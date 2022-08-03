# Create a pytorch-lightning module to encapsulate the transformer model
from typing import Sequence

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from .models import BehaviorCloningClassifier


class BehaviorCloningClassifierModule(pl.LightningModule):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_actions = args.n_actions
        self.model = BehaviorCloningClassifier(input_shape, args)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.metric = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())

    def compute_loss(self, a_hat, a):
        return self.loss_fn(a_hat.view(-1, self.n_actions), a.view(-1))

    def compute_metrics(self, a_hat, a):
        acc = self.metric(a_hat.view(-1, self.n_actions).softmax(dim=-1), a.view(-1))
        self.metric.reset()
        return acc

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, t, y = batch
        y_hat = self(x, t)

        loss = self.compute_loss(y_hat, y)
        self.log("train/loss", loss)
        acc = self.compute_metrics(y_hat, y)
        self.log("train/acc", acc)

        return loss
