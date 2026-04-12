# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .eegnet import EEGNet, EEGNetConfig


@pytest.fixture
def fake_meg():
    batch_size = 2
    n_channels = 4
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [True, False])
def test_eegnet(fake_meg, use_default_config):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_outputs = 3

    if use_default_config:
        config_kwargs = dict()
    else:
        config_kwargs = {
            "n_filters_conv1": 2,
            "n_filters_conv2": 6,
            "conv_kernel_len1": 4,
            "conv_kernel_len3": 6,
            "depth": 1,
            "pool_kernel_len1": 2,
            "pool_kernel_len2": 2,
            "dropout": 0.2,
        }

    model = EEGNetConfig(**config_kwargs).build(n_in_channels, n_outputs)
    assert isinstance(model, EEGNet)

    out = model(fake_meg)
    assert out.shape == (batch_size, n_outputs)
