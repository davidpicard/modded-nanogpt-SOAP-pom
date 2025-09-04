import pytorch_lightning as pl
from data.distributed_loader import DistributedDataLoader
from typing import Optional

class GPTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """Initialize GPT DataModule.
        
        Args:
            cfg: Hydra config object containing data and training parameters
        """
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None

    def prepare_data(self):
        """Called only once and on 1 GPU."""
        # Nothing to prepare as data is already processed into .bin files
        pass

    def setup(self, stage: Optional[str] = None):
        """Set up data loaders for training and validation.
        
        Args:
            stage: either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            # Use trainer.world_size and trainer.global_rank which are automatically set by Lightning
            world_size = self.trainer.world_size if self.trainer else 1
            global_rank = self.trainer.global_rank if self.trainer else 0

            self.train_loader = DistributedDataLoader(
                self.cfg.data.train.input_bin,
                self.cfg.training.batch_size,
                self.cfg.training.sequence_length,
                global_rank,
                world_size
            )
            
            self.val_loader = DistributedDataLoader(
                self.cfg.data.val.input_bin,
                self.cfg.training.batch_size,
                self.cfg.training.sequence_length,
                global_rank,
                world_size
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # Data is already on GPU from DistributedDataLoader
        return batch
