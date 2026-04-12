# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

from ..utils import all_subclasses
from .base import BaseLossConfig
from .losses import ClipLoss, MultiLoss, SigLipLoss

# Find existing loss config subclasses
LossConfig = BaseLossConfig

LossConfig = tp.Annotated[  # type: ignore
    tp.Union[tuple(all_subclasses(BaseLossConfig))],
    pydantic.Field(discriminator="name"),
]


class MultiLossConfig(pydantic.BaseModel):
    losses: list[LossConfig]
    weights: list[float] | None = None

    def build(self):
        losses = {loss.name: loss.build() for loss in self.losses}
        return MultiLoss(losses, self.weights)


def __getattr__(name: str) -> tp.Any:
    if name == "LossConfigSubclasses":
        warnings.warn(
            "LossConfigSubclasses is replaced by LossConfig", DeprecationWarning
        )
        return LossConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
