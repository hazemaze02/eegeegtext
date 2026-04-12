# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchvision.ops import MLP

from .simpleconv import SimpleConvConfig, SimpleConvTimeAggConfig


@pytest.fixture
def fake_meg():
    batch_size = 2
    n_channels = 4
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("time_agg_out", ["gap", "linear", "att"])
def test_simple_conv_time_agg_shape(fake_meg, time_agg_out):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 64

    model_kwargs = {
        "hidden": 64,
        "merger": False,
        "subject_layers": False,
        "time_agg_out": time_agg_out,
    }
    model = SimpleConvTimeAggConfig(**model_kwargs).build(n_in_channels, n_out_channels)
    out = model(fake_meg)
    assert out.shape == (batch_size, n_out_channels)


@pytest.mark.parametrize(
    "depth,merger,subject_layers,initial_linear",
    [
        [4, False, False, 0],
        [6, True, True, 4],
    ],
)
def test_simple_conv_shape(fake_meg, depth, merger, subject_layers, initial_linear):
    batch_size, n_in_channels, n_times = fake_meg.shape
    n_out_channels = 6

    config_kwargs = {
        "hidden": 5,
        "depth": depth,
        "merger": merger,
        "merger_pos_dim": 2048,
        "subject_layers": subject_layers,
        "n_subjects": batch_size,
        "initial_linear": initial_linear,
    }
    model = SimpleConvConfig(**config_kwargs).build(n_in_channels, n_out_channels)

    subject_ids = torch.arange(batch_size)
    channel_positions = torch.randn(batch_size, n_in_channels, 2)

    out = model(fake_meg, subject_ids=subject_ids, channel_positions=channel_positions)
    assert out.shape == (batch_size, n_out_channels, n_times)


def test_simple_conv_different_n_in_channels(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 6

    config_kwargs = {
        "hidden": 5,
        "depth": 2,
        "merger": True,
        "merger_pos_dim": 2048,
        "subject_layers": False,
        "initial_linear": False,
    }
    model = SimpleConvConfig(**config_kwargs).build(n_in_channels, n_out_channels)
    channel_positions = torch.randn(batch_size, n_in_channels, 2)

    out1 = model(fake_meg, channel_positions=channel_positions)
    out2 = model(fake_meg[:, :3, :], channel_positions=channel_positions[:, :3, :])
    assert out1.shape == out2.shape


def test_simple_conv_time_agg_zero_depth_shape(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 64

    model_kwargs = {
        "hidden": 64,
        "depth": 0,
        "merger": False,
        "subject_layers": False,
        "initial_linear": n_out_channels,
        "backbone_out_channels": n_out_channels,
        "time_agg_out": "gap",
        "output_head_config": None,
    }
    model = SimpleConvTimeAggConfig(**model_kwargs).build(n_in_channels, n_out_channels)
    out = model(fake_meg)
    assert out.shape == (batch_size, n_out_channels)


def test_simple_conv_time_agg_output_head(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 6
    head_out = 8

    model_kwargs = {
        "hidden": 5,
        "merger": False,
        "subject_layers": False,
        "time_agg_out": "gap",
        "output_head_config": {
            "hidden_sizes": [head_out],
            "norm_layer": "layer",
            "activation_layer": "relu",
            "dropout": 0.5,
        },
    }
    model = SimpleConvTimeAggConfig(**model_kwargs).build(n_in_channels, n_out_channels)

    out = model(fake_meg)
    assert isinstance(model.output_head, MLP)
    assert out.shape == (batch_size, head_out)


def test_simple_conv_time_agg_output_heads(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 6

    clip_head_out = 8
    mse_head_out = 16

    model_kwargs = {
        "hidden": 16,
        "merger": False,
        "subject_layers": False,
        "time_agg_out": "gap",
        "output_head_config": {
            "clip": {
                "hidden_sizes": [clip_head_out],
                "norm_layer": "layer",
                "activation_layer": "relu",
                "dropout": 0.5,
            },
            "mse": {
                "hidden_sizes": [32, mse_head_out],
                "norm_layer": "instance",
                "activation_layer": "gelu",
                "dropout": 0.0,
            },
        },
    }
    model = SimpleConvTimeAggConfig(**model_kwargs).build(n_in_channels, n_out_channels)

    out = model(fake_meg)
    assert out["clip"].shape == (batch_size, clip_head_out)
    assert out["mse"].shape == (batch_size, mse_head_out)
