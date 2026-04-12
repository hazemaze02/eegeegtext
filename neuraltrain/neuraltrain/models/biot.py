# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from braindecode.models import BIOT
from torch import nn

from .base import BaseModelConfig


class BIOTConfig(BaseModelConfig):
    """Biosignal Transformer from [1] that uses Fourier Transform tokenization of short segments.

    References
    ----------
        [1] Yang, Chaoqi, M. Westover, and Jimeng Sun. "BIOT: Biosignal transformer for cross-data
        learning in the wild." Advances in Neural Information Processing Systems 36 (2024).
    """

    name: tp.Literal["BIOT"] = "BIOT"

    sfreq: float
    emb_size: int = 256
    att_num_heads: int = 8
    n_layers: int = 4
    hop_length: int = 100
    return_feature: bool = False
    chs_info: list[dict] | None = None
    n_times: int | None = None
    input_window_seconds: float | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        kwargs = self.model_dump()
        kwargs.pop("name")
        return BIOT(**kwargs, n_chans=n_in_channels, n_outputs=n_outputs)
