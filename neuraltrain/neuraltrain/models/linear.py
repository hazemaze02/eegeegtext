# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch import nn

from .base import BaseModelConfig
from .common import SubjectLayers


class LinearModelConfig(BaseModelConfig):
    name: tp.Literal["LinearModel"] = "LinearModel"
    reduction: tp.Literal["mean", "concat"] = "mean"
    subject_layers: bool = True
    n_subjects: int = 200

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return LinearModel(
            n_in_channels,
            n_outputs,
            reduction=self.reduction,
            subject_layers=self.subject_layers,
            n_subjects=self.n_subjects,
        )


class LinearModel(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        reduction: str = "mean",
        subject_layers: bool = True,
        n_subjects: int = 200,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_outputs = n_outputs
        self.subject_layers = subject_layers
        self.linear: tp.Any
        if self.subject_layers:
            self.linear = SubjectLayers(
                in_channels=n_in_channels, out_channels=n_outputs, n_subjects=n_subjects
            )
        else:
            self.linear = nn.Linear(n_in_channels, n_outputs)
        self.reduction = reduction

    def forward(self, x, subject_id=None):
        if len(x.shape) > 2:
            if self.reduction == "concat":
                x = x.view(x.size(0), -1)
            elif self.reduction == "mean":
                x = x.mean(dim=-1)
        if self.subject_layers:
            x = self.linear(x.unsqueeze(-1), subject_id).squeeze(-1)
        else:
            x = self.linear(x)
        return x
