import pytorch_lightning as pl
import numpy as np
from torch import Tensor
from torch.optim import Optimizer


class SpikeDetectorCallback(pl.Callback):
    def __init__(
        self,
        buffer_size: int = 25
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = []
        self.do_step = True

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: Tensor) -> None:
        add_loss = True
        if len(self.buffer) >= self.buffer_size:
            mean = np.mean(self.buffer)
            sdev = np.std(self.buffer)
            if loss > mean + 4*sdev:
                print(f"Spike detected! l: {loss:.4f} m: {mean:.4f} s: {sdev:.4f}")
                self.do_step = False
                add_loss = False
        if add_loss:
            self.buffer.append(loss.item())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        if not self.do_step:
            print(f"Spike registered, skipping step for {optimizer.__class__}!")
            optimizer.zero_grad()
        self.do_step = True
