# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from .base import ClipLossConfig, SigLipLossConfig, TorchLossConfig
from .losses import ClipLoss, SigLipLoss


@pytest.fixture
def inputs():
    return torch.randn(8, 4), torch.randn(8, 4)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_mse_loss_config(inputs, reduction: str) -> None:
    loss = TorchLossConfig(name="MSELoss", kwargs={"reduction": reduction}).build()  # type: ignore
    out = loss(*inputs)
    out2 = nn.MSELoss(reduction=reduction)(*inputs)
    assert torch.allclose(out, out2, atol=1e-5)


@pytest.mark.parametrize("weight", [None, [0.1, 0.2, 0.3, 0.4]])
def test_cross_entropy_loss_config(weight) -> None:
    n_examples, n_classes = 8, 4
    y_pred = torch.randn(n_examples, n_classes)
    y_true = torch.arange(n_examples) % n_classes

    kwargs = {
        "weight": weight,
        "ignore_index": 0,
        "reduction": "sum",
        "label_smoothing": 0.1,
    }
    if weight is not None:
        kwargs["weight"] = torch.Tensor(weight)
    loss = TorchLossConfig(name="CrossEntropyLoss", kwargs=kwargs).build()
    out = loss(y_pred, y_true)
    out2 = nn.CrossEntropyLoss(**kwargs)(y_pred, y_true)
    assert torch.allclose(out, out2, atol=1e-5)


def test_clip_loss_config(inputs) -> None:
    kwargs = {"norm_kind": "x", "symmetric": True, "temperature": True}
    loss = ClipLossConfig(**kwargs).build()  # type: ignore
    out = loss(*inputs)
    out2 = ClipLoss(**kwargs)(*inputs)  # type: ignore
    assert out == out2


def test_siglip_loss_config(inputs) -> None:
    kwargs = {"norm_kind": "x", "temperature": True, "bias": True}
    loss = SigLipLossConfig(**kwargs).build()  # type: ignore
    out = loss(*inputs)
    out2 = SigLipLoss(**kwargs)(*inputs)  # type: ignore
    assert out == out2


@pytest.mark.parametrize("name", ["MSELoss", "CrossEntropyLoss"])
@pytest.mark.parametrize("kwargs", [{"reduction": "mean"}, {"reduction": "sum"}])
def test_torchmetrics_config(name, kwargs) -> None:
    x, y = torch.randn(8, 4), torch.randn(8, 4)
    loss = TorchLossConfig(name=name, kwargs=kwargs).build()
    out = loss(x, y)
    out2 = getattr(nn, name)(**kwargs)(x, y)
    assert out == out2


def test_torchloss_config_validation() -> None:
    with pytest.raises(TypeError):
        TorchLossConfig(name="MSELoss", kwargs={"reduction": 12})
