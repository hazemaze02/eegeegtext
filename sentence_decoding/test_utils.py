# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .utils import StandardScaler


def test_standard_scaler():
    batch_size, n_latent_dims = 8, 16
    X = torch.rand(batch_size, n_latent_dims)
    scaler = StandardScaler(dim=1)
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=0), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled_X.std(dim=0), torch.tensor(1.0), atol=1e-6)


def test_standard_scaler_3d():
    batch_size, n_latent_dims, n_times = 64, 768, 300
    X = torch.rand(batch_size, n_latent_dims, n_times)
    scaler = StandardScaler(dim=1)
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=(0, 2)), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled_X.std(dim=(0, 2)), torch.tensor(1.0), atol=1e-6)


def test_standard_scaler_partial_fit():
    batch_size, n_latent_dims, n_times = 64, 768, 300
    X = torch.rand(batch_size, n_latent_dims, n_times)
    scaler = StandardScaler(dim=1)
    scaler.partial_fit(X[: batch_size // 2])
    scaler.partial_fit(X[batch_size // 2 :])
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=(0, 2)), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(scaled_X.std(dim=(0, 2)), torch.tensor(1.0), atol=1e-4)
