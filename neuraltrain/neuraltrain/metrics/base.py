# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for metrics.
"""

import typing as tp

import pydantic
from torch import nn
from torchmetrics import Metric

from neuralset.infra import helpers
from neuraltrain.utils import all_subclasses

from .metrics import ExpectedAccuracy, Rank, TopkAcc

TORCHMETRICS_NAMES = {
    cls.__name__: cls
    for cls in all_subclasses(Metric)
    if cls not in (ExpectedAccuracy, Rank, TopkAcc)
}


class BaseMetricConfig(pydantic.BaseModel):
    """Base class for loss configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")

    log_name: str
    name: str

    def build(self) -> nn.Module:
        raise NotImplementedError


class TorchMetricConfig(BaseMetricConfig):
    name: tp.Literal[tuple(TORCHMETRICS_NAMES.keys())]  # type: ignore
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # validation of mandatory/extra args + basic types (str/int/float)
        helpers.validate_kwargs(TORCHMETRICS_NAMES[self.name], self.kwargs)

    def build(self) -> nn.Module:
        return TORCHMETRICS_NAMES[self.name](**self.kwargs)  # type: ignore


class RankConfig(BaseMetricConfig):
    name: tp.Literal["Rank"] = "Rank"
    reduction: tp.Literal["mean", "median", "std"] = "median"
    relative: bool = False

    def build(self) -> nn.Module:
        return Rank(
            reduction=self.reduction,
            relative=self.relative,
        )


class TopkAccConfig(BaseMetricConfig):
    name: tp.Literal["TopkAcc"] = "TopkAcc"
    topk: int = 1

    def build(self) -> nn.Module:
        return TopkAcc(topk=self.topk)


class ExpectedAccuracyConfig(BaseMetricConfig):
    name: tp.Literal["ExpectedAccuracy"] = "ExpectedAccuracy"

    def build(self) -> nn.Module:
        return ExpectedAccuracy(
            loss=None,
        )
