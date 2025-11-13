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

    for sl in [256, 512, 1024, 2048, 4096, 8192, 16384, 65536]:
        bs = 65536//sl
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                cb = TextGenerationCallback(max_new_tokens=model.model.seq_length//2-1, every_n_steps=1, num_unconditional=cfg.evaluation.num_unconditional_samples)
                # warmup model
                generator = torch.Generator(device="cuda")
                generator.manual_seed(3407)
                prompt = torch.randint(0, model.model.vocab_size, size=(1,1), generator=generator, device="cuda")
                logits, _ = model.model(prompt)
                logits.detach().cpu().numpy().mean()

                # first pass
                print(f"First pass")
                generator = torch.Generator(device="cuda")
                generator.manual_seed(3407)
                n_tokens = sl*bs
                idx = torch.randint(0, model.model.vocab_size, size=(bs, sl), generator=generator, device="cuda")

                start_loop_time = time.perf_counter()
                for i in tqdm(range(cb.num_unconditional)):
                    logits, _ = model.model(idx)
                end_loop_time = time.perf_counter()
                elapsed_loop_time = end_loop_time - start_loop_time
                print(f"1st pass: {bs}x{sl}={n_tokens}tokens in {elapsed_loop_time}s, speed: {n_tokens/elapsed_loop_time} max mem: {torch.cuda.max_memory_allocated()}")


                # second pass
                print(f"Second pass")
                generator = torch.Generator(device="cuda")
                generator.manual_seed(3407)
                n_tokens = sl*bs
                idx = torch.randint(0, model.model.vocab_size, size=(bs, sl), generator=generator, device="cuda")

                start_loop_time = time.perf_counter()
                for i in tqdm(range(cb.num_unconditional)):
                    logits, _ = model.model(idx)
                end_loop_time = time.perf_counter()
                elapsed_loop_time = end_loop_time - start_loop_time
                print(f"2nd pass: {bs}x{sl}={n_tokens}tokens in {elapsed_loop_time}s, speed: {n_tokens/elapsed_loop_time} max mem: {torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()