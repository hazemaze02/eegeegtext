# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

# explicit reimport for API
from .audio import MelSpectrum as MelSpectrum  # noqa
from .audio import Wav2Vec as Wav2Vec  # noqa

# use all features as discriminated
from .base import BaseFeature as BaseFeature
from .base import LabelEncoder as LabelEncoder
from .base import Pulse as Pulse
from .base import Stimulus as Stimulus
from .image import HOG, LBP, RFFT2D, ColorHistogram
from .image import Image as Image  # noqa
from .image import ImageTransformer as ImageTransformer  # noqa
from .neuro import *  # noqa
from .text import *  # noqa
from .video import OpticalFlow as OpticalFlow
from .video import Video as Video  # noqa

FeatureConfig = BaseFeature


def update_config_feature() -> None:
    global FeatureConfig  # pylint: disable=global-statement
    from .base import BaseFeature

    FeatureConfig = tp.Annotated[  # type: ignore
        tp.Union[
            tuple(x for x in BaseFeature._CLASSES.values())
        ],  # if "name" in x.model_fields)],
        pydantic.Field(discriminator="name"),  # serves for pydantic
    ]


update_config_feature()


def __getattr__(name: str) -> tp.Any:
    if name == "CfgFeature":
        warnings.warn("CfgFeature is replaced by FeatureConfig", DeprecationWarning)
        return FeatureConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
