import pytorch_lightning as pl
import torch
import wandb
import tiktoken
from typing import Any, Dict, List

class TextGenerationCallback(pl.Callback):
    def __init__(
        self,
        every_n_steps: int,
        num_unconditional: int = 3,
        num_completions: int = 3,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        prompt_length: int = 32,
        base_seed: int = 42
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_unconditional = num_unconditional
        self.num_completions = num_completions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.prompt_length = prompt_length
        self.base_seed = base_seed
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']

    def _sample_from_model(
        self,
        model: torch.nn.Module,
        prompt_tokens: torch.Tensor = None,
        generator: torch.Generator = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            if prompt_tokens is None:
                tokens = torch.tensor([[self.eot]], dtype=torch.long, device=device)
            else:
                tokens = prompt_tokens.clone().contiguous()

            for _ in range(self.max_new_tokens):
                logits, _ = model(tokens, targets=None, return_logits=True)
                logits = logits[:, -1, :] / self.temperature

                if self.top_k is not None:
                    v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                tokens = torch.cat([tokens, next_token], dim=1).contiguous()

                if tokens.shape[1] > 1 and next_token.item() == self.eot:
                    break

        model.train()
        return tokens

    def _generate_samples(
        self,
        pl_module: pl.LightningModule,
        val_loader: Any,
        device: str = 'cuda'
    ) -> Dict[str, List[str]]:
        samples = {'unconditional': [], 'completions': []}

        # Unconditional samples
        for i in range(self.num_unconditional):
            generator = torch.Generator(device=device)
            generator.manual_seed(self.base_seed + i)
            
            tokens = self._sample_from_model(
                pl_module.model,
                prompt_tokens=None,
                generator=generator,
                device=device
            )
            text = self.enc.decode(tokens[0].cpu().numpy())
            samples['unconditional'].append(text)

        # Reset validation loader for consistent prompts
        val_loader.reset()
        val_x, _ = val_loader.next_batch()
        completion_indices = list(range(min(self.num_completions, val_x.shape[0])))

        for i in completion_indices:
            prompt_tokens = val_x[i:i+1, :self.prompt_length]
            generator = torch.Generator(device=device)
            generator.manual_seed(self.base_seed + 1000 + i)

            completion_tokens = self._sample_from_model(
                pl_module.model,
                prompt_tokens=prompt_tokens,
                generator=generator,
                device=device
            )

            prompt_text = self.enc.decode(prompt_tokens[0].cpu().numpy())
            full_text = self.enc.decode(completion_tokens[0].cpu().numpy())
            completion_text = full_text[len(prompt_text):]

            samples['completions'].append({
                'prompt': prompt_text,
                'completion': completion_text,
                'full': full_text
            })

        return samples

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        if not trainer.is_global_zero:
            return

        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            samples = self._generate_samples(
                pl_module,
                trainer.datamodule.val_loader,
                device=pl_module.device
            )

            # Create tables for samples
            unconditional_table = wandb.Table(
                columns=["sample_id", "generated_text"]
            )
            for i, text in enumerate(samples['unconditional']):
                unconditional_table.add_data(i+1, text)

            completion_table = wandb.Table(
                columns=["sample_id", "prompt", "completion", "full_text"]
            )
            for i, sample in enumerate(samples['completions']):
                completion_table.add_data(
                    i+1,
                    sample['prompt'],
                    sample['completion'],
                    sample['full']
                )

            # Log to wandb
            wandb.log({
                "step": step,
                "unconditional_samples": unconditional_table,
                "completion_samples": completion_table
            })

            # Clear GPU memory
            torch.cuda.empty_cache()

class WandBLoggingCallback(pl.Callback):
    def __init__(self, log_every_n_steps: int = 1):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._total_val_loss = 0.0
        self._val_steps = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        if not trainer.is_global_zero:
            return

        step = trainer.global_step
        if step > 0 and step % self.log_every_n_steps == 0:
            # Get learning rate from optimizer
            if hasattr(trainer.optimizers[0], 'param_groups'):
                current_lr = trainer.optimizers[0].param_groups[0]['lr']
            else:
                current_lr = trainer.optimizers[0].defaults['lr']

            # Log metrics
            wandb.log({
                "train_loss": outputs["loss"],
                "learning_rate": current_lr,
                "step": step
            })

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        if not trainer.is_global_zero:
            return
        
        # Accumulate validation loss
        self._total_val_loss += outputs
        self._val_steps += 1

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return

        # Log average validation loss
        if self._val_steps > 0:
            avg_val_loss = self._total_val_loss / self._val_steps
            wandb.log({
                "val_loss": avg_val_loss,
                "step": trainer.global_step
            })

            # Reset accumulators
            self._total_val_loss = 0.0
            self._val_steps = 0

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
            
        # Log model parameter counts at the start of training
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/memory_mb_fp32": total_params * 4 / 1024**2,
            "model/memory_mb_bf16": total_params * 2 / 1024**2
        })
