# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torchvision.ops import MLP

from .common import (
    ChannelMerger,
    Mean,
    MlpConfig,
    NormDenormScaler,
    SubjectLayers,
    UnitNorm,
)


@pytest.fixture
def fake_meg() -> torch.Tensor:
    batch_size = 4
    n_channels = 8
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("init_id,out_channels", [[False, 7], [True, 8], [True, 9]])
def test_subject_layers(fake_meg: torch.Tensor, init_id: bool, out_channels: int) -> None:
    batch_size, in_channels, n_times = fake_meg.shape
    n_subjects = 4
    subjects = torch.arange(0, batch_size) % n_subjects

    if init_id and in_channels != out_channels:
        with pytest.raises(ValueError):
            layer = SubjectLayers(in_channels, out_channels, n_subjects, init_id)
    else:
        layer = SubjectLayers(in_channels, out_channels, n_subjects, init_id)
        out = layer(fake_meg, subjects)
        assert out.shape == (batch_size, out_channels, n_times)


def test_subject_layers_invalid_subject(fake_meg: torch.Tensor) -> None:
    batch_size, in_channels, _ = fake_meg.shape
    n_subjects = 1
    layer = SubjectLayers(in_channels, 2, n_subjects, init_id=False)

    subjects = torch.ones(batch_size)
    with pytest.raises(AssertionError):
        layer(fake_meg, subjects)


@pytest.mark.parametrize(
    "dropout,usage_penalty,per_subject", [[0.0, 0.0, False], [0.5, 0.5, True]]
)
def test_channel_merger_shape(fake_meg, dropout, usage_penalty, per_subject):
    chout = 8
    pos_dim = 2048  # XXX Doesn't work for all numbers! E.g., 256 will fail

    batch_size, n_in_channels, n_times = fake_meg.shape
    positions = torch.randn(batch_size, n_in_channels, 2)
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        chout,
        pos_dim,
        dropout=dropout,
        usage_penalty=usage_penalty,
        n_subjects=10,
        per_subject=per_subject,
    )
    out = merger(fake_meg, subject_ids, positions)

    assert out.shape == (batch_size, chout, n_times)


@pytest.mark.parametrize("affine", [True, False])
def test_norm_denorm_scaler(affine: bool) -> None:
    n_feats = 10
    x1 = torch.rand(100, n_feats)
    x2 = torch.rand(20, n_feats)
    scaler = NormDenormScaler(x1, affine=affine)

    out = scaler(x2)

    if affine:
        assert torch.allclose(out.mean(dim=0), x1.mean(dim=0))
        assert torch.allclose(out.std(dim=0, correction=0), x1.std(dim=0, correction=0))
        assert torch.allclose(out.mean(dim=0), scaler.scaler.bias)
        assert torch.allclose(out.std(dim=0, correction=0), scaler.scaler.weight)
    else:
        torch.allclose(out.mean(dim=0), torch.zeros(n_feats))
        torch.allclose(out.mean(dim=0), torch.ones(n_feats))


@pytest.mark.skip(reason="TODO")
def test_bahdanau_attention():
    raise NotImplementedError


def test_unit_norm() -> None:
    x = torch.rand(10, 16)
    norm = UnitNorm()
    y = norm(x)
    assert torch.allclose(y.norm(dim=-1), torch.tensor(1.0))


@pytest.mark.parametrize("n_layers", [1, 3, 0])
@pytest.mark.parametrize("hidden_size", [2, 4])
@pytest.mark.parametrize("norm_kind", [None, "layer", "batch", "instance", "unit"])
@pytest.mark.parametrize("nonlin_kind", [None, "gelu", "relu", "elu", "prelu"])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_mlp(n_layers, hidden_size, norm_kind, nonlin_kind, dropout) -> None:
    batch_size = 32
    input_size = 8
    output_size = 16
    x = torch.rand(batch_size, input_size)

    mlp = MlpConfig(
        hidden_sizes=[hidden_size] * n_layers,
        norm_layer=norm_kind,
        activation_layer=nonlin_kind,
        dropout=dropout,
    ).build(
        input_size,
        output_size,
    )

    assert isinstance(mlp, (MLP, nn.Identity))
    out = mlp(x)
    assert out.shape == (batch_size, output_size if n_layers > 0 else input_size)


@pytest.mark.parametrize("output_size", [None, 2])
def test_mlp_input_output_sizes(output_size) -> None:
    batch_size = 32
    input_size = 8
    hidden_size = 16
    x = torch.rand(batch_size, input_size)

    mlp = MlpConfig(
        input_size=input_size,
        hidden_sizes=[hidden_size] * 3,
    ).build(
        output_size=output_size,
    )

    out = mlp(x)
    assert out.shape == (batch_size, hidden_size if output_size is None else output_size)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(dim: int, keepdim: bool) -> None:
    x = torch.rand((10, 9, 8))
    mean_layer = Mean(dim=dim, keepdim=keepdim)
    out = mean_layer(x)
    assert torch.allclose(out, x.mean(dim=dim, keepdim=keepdim))
