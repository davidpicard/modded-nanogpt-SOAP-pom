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

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.utils import make_table
from tqdm import tqdm
import torch.nn.functional as F
import tiktoken

class MyCustomLM(LM):
    def __init__(self, model, temperature=1., top_k=50):
        self.model = model
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        self.temperature = temperature
        self.top_k = top_k
        self._rank = 0
        self._world_size = 1

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        answers = []
        with torch.no_grad():
            for r in tqdm(requests):
                tok_in = torch.tensor([self.enc.encode(r.args[0])], dtype=torch.long, device='cuda')
                tok_out = torch.tensor([self.enc.encode(r.args[1])], dtype=torch.long, device='cuda')
                n = tok_in.shape[1]
                m = tok_out.shape[1]
                tokens = torch.cat([tok_in, tok_out], dim=-1)
                logits, _ = self.model(tokens, targets=tokens, return_logits=True)
                m = F.log_softmax(logits[:, n:n + m, :], dim=-1).cpu().max(dim=-1)
                ll = m.values.sum()
                is_greedy = (m.indices == tok_out.cpu()).all()
                answers.append((ll, is_greedy))
        return answers

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        answers = []
        with torch.no_grad():
            for r in requests:
                tok_in = torch.tensor([self.enc.encode(r.args[0])], dtype=torch.long, device='cuda')
                eot = torch.tensor([[self.eot]], dtype=torch.long, device='cuda')
                n = tok_in.shape[1]
                tokens = torch.cat([eot, tok_in], dim=-1)
                logits, _ = self.model(tokens, targets=tokens, return_logits=True)
                m = F.log_softmax(logits[:, 0:n, :], dim=-1).cpu().max(dim=-1)
                ll = m.values
                answers.append(([ll],))
        return answers

    def generate_until(self, requests: list[Instance]) -> list[str]:
        answers = []
        print("generate until")
        for r in requests:
            s, d = r.args
            self.model.eval()
            with torch.no_grad():
                if s is None:
                    tokens = torch.tensor([[self.eot]], dtype=torch.long, device='cuda')
                else:
                    s = torch.tensor([self.enc.encode(s)], dtype=torch.long, device='cuda')
                    tokens = s.clone().contiguous()

                state = self.model.reset(batch_size=1)
                idx = tokens
                for _ in range(d['max_gen_toks']):
                    logits, state = self.model.ar_forward(idx=idx, state=state)
                    logits = logits[:, -1, :] / self.temperature

                    if self.top_k is not None:
                        v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('inf')

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1, generator=None)
                    tokens = torch.cat([tokens, next_token], dim=1).contiguous()

                    idx = next_token.view(1, 1)
                    if tokens.shape[1] > 1 and next_token.item() == self.eot:
                        break

            self.model.train()
            s = self.enc.decode(tokens[0].cpu().numpy())
            answers.append(s)
        return s

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function using PyTorch Lightning."""
    print(f"Running pytorch {torch.version.__version__}")


    # Initialize model  from ckpt
    mod = instantiate(cfg.model.gpt)
    my_model = LitGPT(model=mod, cfg=cfg)

    print(f"loading ckpt from {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    my_model.load_state_dict(ckpt['state_dict'])
    print(f"ckpt loaded from {cfg.ckpt_path}")
    ckpt = None

    my_model = my_model.model.to("cuda")
    if cfg.hardware.compile:
        print(f"Compiling model!")
        my_model.compile()
    my_model.eval()

    custom_model = MyCustomLM(my_model)
    limit = None
    if hasattr(cfg, 'limit'):
        limit = cfg.limit
        print(f"limiting to {limit}")
    summary = []

    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=custom_model,
        tasks=["arc_easy"],
        num_fewshot=25,
        batch_size=1,
        limit=limit,
    )
    summary.append(make_table(results))
    print(results['results'])
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=custom_model,
        tasks=["hellaswag"],
        num_fewshot=10,
        batch_size=1,
        limit=limit,
    )
    summary.append(make_table(results))
    print(results['results'])
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=custom_model,
        tasks=["winogrande"],
        num_fewshot=5,
        batch_size=1,
        limit=limit,
    )
    summary.append(make_table(results))
    print(results['results'])
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=custom_model,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=1,
        limit=limit,
    )
    summary.append(make_table(results))
    print(results['results'])

    for s in summary:
        print(s)

if __name__ == "__main__":
    main()