# Datamodule to load offline training data
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import OfflineTrajectoryDataset


class OfflineTrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: Optional[int] = 16,
        num_workers: Optional[int] = 2,
        pin_memory: Optional[bool] = torch.cuda.is_available(),
        **kwargs
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        self.train_dataset = OfflineTrajectoryDataset(self.root_dir, **self.kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
