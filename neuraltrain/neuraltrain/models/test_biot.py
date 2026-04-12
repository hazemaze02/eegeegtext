# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from braindecode.models import BIOT

from .biot import BIOTConfig


@pytest.fixture
def fake_eeg():
    batch_size = 2
    n_channels = 4
    n_times = 200
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [False, True])
def test_biot(fake_eeg, use_default_config):
    batch_size, n_in_channels, n_times = fake_eeg.shape
    n_outputs = 3

    if use_default_config:
        config_kwargs = {
            "sfreq": 200.0,
        }
    else:
        config_kwargs = {
            "sfreq": 200.0,
            "emb_size": 256,
            "att_num_heads": 8,
            "n_layers": 4,
            "hop_length": 100,
            "return_feature": False,
            "chs_info": None,
            "n_times": n_times,
            "input_window_seconds": None,
        }

    model = BIOTConfig(**config_kwargs).build(n_in_channels, n_outputs)
    assert isinstance(model, BIOT)

    out = model(fake_eeg)
    assert out.shape == (batch_size, n_outputs)
