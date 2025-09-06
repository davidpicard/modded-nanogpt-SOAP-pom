import os
import uuid
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
import wandb
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from lit_module import LitGPT
from lit_data import GPTDataModule
from callbacks import TextGenerationCallback, WandBLoggingCallback
from callbacks.hellaswag import HellaSwagCallback
from callbacks.spike_detector import SpikeDetectorCallback

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function using PyTorch Lightning."""
    print(f"Running pytorch {torch.version.__version__}")
    
    # Initialize wandb (only on rank 0)
    # if cfg.trainer.devices == 1 or (hasattr(cfg.trainer, 'global_rank') and cfg.trainer.global_rank == 0):
    rank_zero_only(wandb.init)(
        project="pom_archi",
        config=dict(cfg),
        name=cfg.experiment_name
    )


    # Initialize callbacks
    callbacks = [
        # SpikeDetectorCallback(),
        TextGenerationCallback(
            every_n_steps=cfg.evaluation.sample_every,
            num_unconditional=cfg.evaluation.num_unconditional_samples,
            num_completions=cfg.evaluation.num_completion_samples,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            temperature=cfg.evaluation.temperature,
            top_k=cfg.evaluation.top_k,
            prompt_length=cfg.evaluation.prompt_length,
            base_seed=cfg.evaluation.sample_seed
        ),
        HellaSwagCallback(
            every_n_steps=cfg.evaluation.hellaswag_every,
            base_seed=3407
        ),
        WandBLoggingCallback(log_every_n_steps=cfg.logging.log_every),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(cfg.logging.log_dir, str(uuid.uuid4())),
            filename='gpt-{step:06d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            every_n_train_steps=cfg.evaluation.save_every
        )
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.num_iterations,
        callbacks=callbacks,
        strategy=DDPStrategy(
            process_group_backend=cfg.distributed.backend,
            find_unused_parameters=cfg.distributed.find_unused_parameters
        ),
        precision=16 if cfg.hardware.dtype == 'float16' else 32,
        **cfg.trainer
    )

    # Initialize model and data module
    model = LitGPT(instantiate(cfg.model.gpt), cfg=cfg)
    if cfg.hardware.compile:
        print(f"Compiling model!")
        model.model.compile()
    data_module = GPTDataModule(cfg)
    # Train
    trainer.fit(model, data_module)

    # Clean up
    wandb.finish()

if __name__ == "__main__":
    main()