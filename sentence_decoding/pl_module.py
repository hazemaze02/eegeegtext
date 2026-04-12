# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn, optim

from neuralset.dataloader import SegmentData


class BrainModule(pl.LightningModule):
    """
    Torch-lightning module for M/EEG model training.
    """

    def __init__(
        self,
        model,
        transformer,
        loss,
        metrics,
        retrieval_metrics,
        trainer_config,
        target_scaler=None,
        checkpoint_path=None,
    ):
        super().__init__()
        self.model = model
        self.transformer = transformer
        self.trainer_config = trainer_config

        self.target_scaler = target_scaler
        self.checkpoint_path = checkpoint_path

        self.loss = loss
        self.metrics = nn.ModuleDict(
            {split + "_" + k: v for k, v in metrics.items() for split in ["val", "test"]}
        )
        self.retrieval_metrics = nn.ModuleDict(
            {
                split + "_" + k: v
                for k, v in retrieval_metrics.items()
                for split in ["val", "test"]
            }
        )

    def cnn_forward(self, batch):
        x = batch.data["neuro"]

        subject_ids = batch.data["subject_id"] if "subject_id" in batch.data else None
        channel_positions = (
            batch.data["channel_positions"] if "channel_positions" in batch.data else None
        )

        model_name = self.model.__class__.__name__
        if "SimpleConv" in model_name:
            y_pred = self.model(x, subject_ids, channel_positions)
        elif model_name == "EEGNet":
            y_pred = self.model(x)
        elif model_name in ["LinearModel"]:
            y_pred = self.model(x, subject_ids)
        else:
            raise ValueError(f"Unknown model {model_name}")

        return y_pred

    def transformer_forward(self, batch, y_pred):
        sentence_uids = np.array(
            [
                f"{segment._trigger['sequence_id']}_{segment._trigger['timeline']}"
                for segment in batch.segments
            ]
        )

        # pad and group according to sentences
        unique_uids, sentence_idx = np.unique(sentence_uids, return_index=True)
        unique_uids = unique_uids[
            np.argsort(sentence_idx)
        ]  # beware of order in np.unique!!!
        grouped_y_pred = []
        for uid in unique_uids:
            indices = [i for i, s in enumerate(sentence_uids) if s == uid]
            grouped_y_pred.append(torch.stack([y_pred[i] for i in indices]))
        max_len = max([len(y) for y in grouped_y_pred])

        # pad for transformer
        transformer_input = torch.zeros(len(grouped_y_pred), max_len, y_pred.shape[1]).to(
            y_pred.device
        )
        mask = torch.zeros(len(grouped_y_pred), max_len).to(y_pred.device)
        for i, y in enumerate(grouped_y_pred):
            transformer_input[i, : len(y)] = y
            mask[i, : len(y)] = 1

        # feed to transformer
        transformer_output = self.transformer(transformer_input, mask=mask.bool())

        # unpad and ungroup
        out = []
        for i, y in enumerate(grouped_y_pred):
            out.extend(transformer_output[i][: len(y)])
        out = torch.stack(out)

        out = out / out.norm(dim=1, keepdim=True)

        return out

    def _run_step(self, batch, step_name):

        y_true = batch.data["feature"]
        if self.target_scaler is not None:
            y_true = self.target_scaler.transform(y_true)

        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_true.shape[0],
        }

        # SimpleConv processing
        y_pred = self.cnn_forward(batch)
        if len(y_pred.shape) == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_true = y_true.reshape(y_true.shape[0], -1)

        y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
        loss = self.loss(y_pred, y_true)
        self.log(f"{step_name}_cnn_loss", loss, **log_kwargs)

        # Transformer processing
        if (
            self.transformer is not None
            and self.current_epoch >= self.trainer_config.transformer_start_epoch
        ):
            y_transformer = self.transformer_forward(batch, y_pred)
            transformer_loss = self.loss(y_transformer, y_true)

            self.log(f"{step_name}_transformer_loss", transformer_loss, **log_kwargs)
            loss = transformer_loss
            y_pred = y_transformer

        # Compute metrics
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(metric_name, metric, **log_kwargs)

        return loss, y_pred, y_true

    def training_step(self, batch: SegmentData, batch_idx: int, dataloader_idx: int = 0):
        loss, _, _ = self._run_step(batch, step_name="train")
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, step_name="val")
        return y_pred, y_true

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, step_name="test")
        return y_pred, y_true

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.trainer_config.lr,
            weight_decay=self.trainer_config.weight_decay,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer_config.n_epochs
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
