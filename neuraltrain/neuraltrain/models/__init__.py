# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

from ..utils import all_subclasses
from .base import BaseModelConfig
from .biot import BIOT
from .eegnet import EEGNet, EEGNetConfig
from .linear import LinearModel, LinearModelConfig
from .simpleconv import (
    SimpleConv,
    SimpleConvConfig,
    SimpleConvTimeAgg,
    SimpleConvTimeAggConfig,
)
from .transformer import TransformerEncoder, TransformerEncoderConfig

# Find existing model config subclasses
ModelConfig = BaseModelConfig

ModelConfig = tp.Annotated[  # type: ignore
    tp.Union[tuple(all_subclasses(BaseModelConfig))],
    pydantic.Field(discriminator="name"),
]


def __getattr__(name: str) -> tp.Any:
    if name == "ModelConfigSubclasses":
        warnings.warn(
            "ModelConfigSubclasses is replaced by ModelConfig", DeprecationWarning
        )
        return ModelConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
