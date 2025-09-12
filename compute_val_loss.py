import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import sys

import pytorch_lightning as pl
from tqdm import tqdm

from lit_data import GPTDataModule
from lit_module import LitGPT
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function using PyTorch Lightning."""
    print(f"Running pytorch {torch.version.__version__}")


    # Initialize model and data module
    mod = instantiate(cfg.model.gpt)
    model = LitGPT.load_from_checkpoint(cfg.evaluation.checkpoint, model=mod)
    model.eval()

    # if cfg.hardware.compile:
    #     print(f"Compiling model!")
    #     model.model.compile()
    data_module = GPTDataModule(cfg)
    data_module.setup()

    # Initialize trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.num_iterations,
        precision=cfg.hardware.precision,
        **cfg.trainer
    )

    trainer.validate(model=model, datamodule=data_module, verbose=True)


if __name__ == "__main__":
    main()