# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""

import typing as tp

from torch import nn

from .base import BaseModelConfig


class EEGNetConfig(BaseModelConfig):
    name: tp.Literal["EEGNet"] = "EEGNet"

    n_filters_conv1: int = 8
    n_filters_conv2: int = 16
    conv_kernel_len1: int = 64
    conv_kernel_len3: int = 16
    depth: int = 2
    pool_kernel_len1: int = 4
    pool_kernel_len2: int = 8
    dropout: float = 0.5

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return EEGNet(n_in_channels, n_outputs, config=self)


class EEGNet(nn.Module):
    """EEGNet compact ConvNet architecture for EEG from [1].

    See Table 2 in EEGNet paper.

    References
    ----------
        [1] Lawhern, Vernon J., et al. "EEGNet: a compact convolutional neural network for
        EEG-based brain–computer interfaces." Journal of neural engineering 15.5 (2018): 056013.

    XXX Implement clamping of weights!
    """

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        config: EEGNetConfig | None = None,
    ):
        super().__init__()
        self.n_outputs = n_outputs  # For easy input/output size characterization outside
        config = config if config is not None else EEGNetConfig()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                config.n_filters_conv1,
                kernel_size=(1, config.conv_kernel_len1),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(config.n_filters_conv1),
            nn.Conv2d(
                config.n_filters_conv1,
                config.depth * config.n_filters_conv1,
                kernel_size=(n_in_channels, 1),
                bias=False,
                groups=config.n_filters_conv1,
            ),
            nn.BatchNorm2d(config.depth * config.n_filters_conv1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, config.pool_kernel_len1)),
            nn.Dropout(config.dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                config.depth * config.n_filters_conv1,
                config.depth * config.n_filters_conv1,
                kernel_size=(1, config.conv_kernel_len3),
                bias=False,
                groups=config.depth * config.n_filters_conv1,
                padding="same",
            ),
            nn.Conv2d(
                config.depth * config.n_filters_conv1,
                config.n_filters_conv2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(config.n_filters_conv2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, config.pool_kernel_len2)),
            nn.Dropout(config.dropout),
            nn.Flatten(1),
        )
        self.classifier = nn.LazyLinear(n_outputs)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B, 1, C, T)

        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)

        return x
