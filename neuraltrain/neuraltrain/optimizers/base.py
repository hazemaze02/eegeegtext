# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for optimizers.
"""

import typing as tp

import pydantic
import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from neuralset.infra import helpers
from neuraltrain.utils import all_subclasses

TORCH_OPTIMIZER_NAMES = [
    cls.__name__ for cls in all_subclasses(Optimizer) if cls.__name__ != "NewCls"
]


class BaseOptimizerConfig(pydantic.BaseModel):
    """Base class for loss configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        raise NotImplementedError


class TorchOptimizerConfig(BaseOptimizerConfig):
    name: tp.Literal[tuple(TORCH_OPTIMIZER_NAMES)]  # type: ignore
    lr: float
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        assert (
            "lr" not in self.kwargs
        ), "lr should be defined as a base parameter instead of within kwargs."
        # validation of mandatory/extra args + basic types (str/int/float)
        helpers.validate_kwargs(getattr(optim, self.name), self.kwargs | {"params": None})

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        return getattr(optim, self.name)(params, lr=self.lr, **self.kwargs)  # type: ignore
