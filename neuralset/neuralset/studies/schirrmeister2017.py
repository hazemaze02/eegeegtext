# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

from .moabb import MOABBDataset2024


class Schirrmeister2017(MOABBDataset2024):
    dataset_name: tp.ClassVar[str] = "Schirrmeister2017"
    description: tp.ClassVar[
        str
    ] = """
    14 subjects performed 1040 trials of 4-second executed movements (left hand, right hand, feet or rest).
    """

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError("Dataset not available to download yet.")
