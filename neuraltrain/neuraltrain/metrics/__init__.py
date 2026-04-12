# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

from ..utils import all_subclasses
from .base import BaseMetricConfig
from .metrics import Rank, TopkAcc

# Find existing metric config subclasses
MetricConfig = BaseMetricConfig

MetricConfig = tp.Annotated[  # type: ignore
    tp.Union[tuple(all_subclasses(BaseMetricConfig))],
    pydantic.Field(discriminator="name"),
]


def __getattr__(name: str) -> tp.Any:
    if name == "MetricConfigSubclasses":
        warnings.warn(
            "MetricConfigSubclasses is replaced by MetricConfig", DeprecationWarning
        )
        return MetricConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
