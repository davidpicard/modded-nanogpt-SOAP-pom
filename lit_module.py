import pytorch_lightning as pl
import torch
import numpy as np

class LitGPT(pl.LightningModule):
    def __init__(self, model, cfg):
        """
        Args:
            model: GPT model instantiated by hydra
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.cfg = cfg
        self.loss_buffer = []
        self.len_buffer = 50

    def forward(self, idx, targets=None, return_logits=False):
        return self.model(idx, targets, return_logits)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        if len(self.loss_buffer) >= self.len_buffer:
            m = np.mean(self.loss_buffer)
            s = np.std(self.loss_buffer)
            if loss.item() > m+4.*s:
                print(f"loss spike: {loss.item()} m: {m} s: {s}")
                loss = 0.*loss + m

            self.log('m', m, prog_bar=True, sync_dist=True)
            self.log('s', s, prog_bar=True, sync_dist=True)
        self.loss_buffer.append(loss.item())
        if len(self.loss_buffer) > self.len_buffer:
            self.loss_buffer.pop(0)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Get list of optimizers from model
        optimizer = self.model.configure_optimizers(
            weight_decay=self.cfg.training.weight_decay,
            learning_rate=self.cfg.training.learning_rate,
            betas=(0.9, 0.9999)
        )
        
        opt_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: self._get_lr_scale(step)
                ),
                'interval': 'step',
                'frequency': 1
            }
        }

        return opt_dict
    
    def _get_lr_scale(self, step: int) -> float:
        """Calculate learning rate scale based on warmup and warmdown."""
        if step < self.cfg.training.warmup_iters:
            return float(step) / float(max(1, self.cfg.training.warmup_iters))
        elif step > self.cfg.training.num_iterations - self.cfg.training.warmdown_iters:
            decay_ratio = float(
                self.cfg.training.num_iterations - step
            ) / float(self.cfg.training.warmdown_iters)
            return max(0.0, decay_ratio)
        return 1.0
