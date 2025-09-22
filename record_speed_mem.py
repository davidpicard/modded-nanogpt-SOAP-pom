import time

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import sys

import pytorch_lightning as pl
from tqdm import tqdm

from callbacks import HellaSwagCallback, TextGenerationCallback
from lit_data import GPTDataModule
from lit_module import LitGPT
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function using PyTorch Lightning."""
    print(f"Running pytorch {torch.version.__version__}")


    # Initialize model and data module
    mod = instantiate(cfg.model.gpt)
    model = LitGPT(model=mod, cfg=cfg)
    model = model.to("cuda")
    if cfg.hardware.compile:
        print(f"Compiling model!")
        model.model.compile()
    model.eval()

    cb = TextGenerationCallback(max_new_tokens=1023, every_n_steps=1)
    # warmup model
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3407)
    tokens = cb._sample_from_model(model, None, generator, "cuda")
    tokens = tokens[0].cpu().numpy()

    # first pass
    print(f"First pass")
    start_loop_time = time.perf_counter()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3407)
    n_tokens = 0

    for i in tqdm(range(cb.num_unconditional)):
        tokens = cb._sample_from_model(model, None, generator, "cuda")
        tokens = tokens[0].cpu().numpy()
        n_tokens += len(tokens)
    end_loop_time = time.perf_counter()
    elapsed_loop_time = end_loop_time - start_loop_time
    print(f"1st pass: {n_tokens} in {elapsed_loop_time}s, speed: {n_tokens/elapsed_loop_time} max mem: {torch.cuda.max_memory_allocated()}")


    # second pass
    print(f"Second pass")
    start_loop_time = time.perf_counter()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3407)
    n_tokens = 0
    for i in tqdm(range(cb.num_unconditional)):
        tokens = cb._sample_from_model(model, None, generator, "cuda")
        tokens = tokens[0].cpu().numpy()
        n_tokens += len(tokens)
    end_loop_time = time.perf_counter()
    elapsed_loop_time = end_loop_time - start_loop_time
    print(f"2nd pass: {n_tokens} in {elapsed_loop_time}s, speed: {n_tokens/elapsed_loop_time}  max mem: {torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()