# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn, optim

from .base import TorchOptimizerConfig


def test_torch_optimizer_config() -> None:
    model = nn.Linear(1, 1)

    optimizer = TorchOptimizerConfig(
        name="Adam",
        lr=1e-5,
        kwargs={
            "betas": (0.1, 0.111),
            "weight_decay": 0.1,
        },
    ).build(
        model.parameters()
    )  # type: ignore

    assert isinstance(optimizer, optim.Adam)
