# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .transformer import TransformerEncoder, TransformerEncoderConfig


@pytest.fixture
def fake_sequence():
    batch_size = 2
    dim = 64
    n_times = 10
    seq = torch.randn(batch_size, n_times, dim)
    return seq


def test_transformer(fake_sequence):
    batch_size, n_times, dim = fake_sequence.shape

    model = TransformerEncoderConfig().build(dim)
    assert isinstance(model, TransformerEncoder)

    out = model(fake_sequence)
    assert out.shape == (batch_size, n_times, dim)
