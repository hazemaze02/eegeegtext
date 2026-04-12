# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom lightning module that wraps a pytorch model.
"""

from pathlib import Path

import lightning.pytorch as pl
from torch import nn
from torchmetrics import Metric

from neuralset.dataloader import SegmentData
from neuraltrain.optimizers import OptimizerConfig


class BrainModule(pl.LightningModule):
    """Torch-lightning module for M/EEG model training."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim_config: OptimizerConfig,
        metrics: dict[str, Metric],
        x_name: str = "input",
        y_name: str = "target",
        max_epochs: int = 100,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_name, self.y_name = x_name, y_name
        self.checkpoint_path = checkpoint_path

        # Optimizer
        self.optim_config = optim_config
        self.max_epochs = max_epochs

        self.loss = loss
        self.metrics = nn.ModuleDict(
            {split + "_" + k: v for k, v in metrics.items() for split in ["val", "test"]}
        )

    def forward(self, batch: SegmentData):
        return self.model(batch.data[self.x_name])

    def _run_step(self, batch: SegmentData, batch_idx, step_name):
        y_true = batch.data[self.y_name]
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=True if step_name == "train" else False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=y_pred.shape[0],
        )

        # Compute metrics
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(
                    metric_name,
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=y_pred.shape[0],
                )

        return loss, y_pred, y_true

    def training_step(self, batch: SegmentData, batch_idx):
        loss, _, _ = self._run_step(batch, batch_idx, step_name="train")
        return loss

    def validation_step(self, batch: SegmentData, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="val")
        return y_pred, y_true

    def test_step(self, batch: SegmentData, batch_idx):
        self._run_step(batch, batch_idx, step_name="test")

    def configure_optimizers(self):
        return self.optim_config.build(self.parameters())
