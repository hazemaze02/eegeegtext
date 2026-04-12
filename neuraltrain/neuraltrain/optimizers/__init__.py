# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic

from ..utils import all_subclasses
from .base import BaseOptimizerConfig

# Find existing optimizer config subclasses
OptimizerConfig = BaseOptimizerConfig

OptimizerConfig = tp.Annotated[  # type: ignore
    tp.Union[tuple(all_subclasses(BaseOptimizerConfig))],
    pydantic.Field(discriminator="name"),
]
