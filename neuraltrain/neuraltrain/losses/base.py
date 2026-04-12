# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for loss functions.
"""

import typing as tp

import pydantic
from torch import nn
from torch.nn.modules.loss import _Loss

from neuralset.infra import helpers
from neuraltrain.utils import all_subclasses

from .losses import ClipLoss, SigLipLoss

TORCHLOSS_NAMES = [cls.__name__ for cls in all_subclasses(_Loss)]


class BaseLossConfig(pydantic.BaseModel):
    """Base class for loss configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self) -> nn.Module:
        raise NotImplementedError


class ClipLossConfig(BaseLossConfig):
    name: tp.Literal["Clip"] = "Clip"
    norm_kind: str | None = "y"
    temperature: bool = False
    symmetric: bool = False

    def build(self) -> nn.Module:
        return ClipLoss(
            norm_kind=self.norm_kind,
            temperature=self.temperature,
            symmetric=self.symmetric,
        )


class SigLipLossConfig(BaseLossConfig):
    name: tp.Literal["SigLip"] = "SigLip"
    norm_kind: str | None = "y"
    temperature: bool = True
    bias: bool = True
    identical_candidates_threshold: float | None = 0.999
    reweigh_positives: bool = True

    def build(self) -> nn.Module:
        return SigLipLoss(
            norm_kind=self.norm_kind,
            temperature=self.temperature,
            bias=self.bias,
            identical_candidates_threshold=self.identical_candidates_threshold,
            reweigh_positives=self.reweigh_positives,
        )


class TorchLossConfig(BaseLossConfig):
    name: tp.Literal[tuple(TORCHLOSS_NAMES)]  # type: ignore
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # validation of mandatory/extra args + basic types (str/int/float)
        helpers.validate_kwargs(getattr(nn, self.name), self.kwargs)

    def build(self) -> nn.Module:
        return getattr(nn, self.name)(**self.kwargs)  # type: ignore
