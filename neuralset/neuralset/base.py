# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pydantic
import yaml
from typing_extensions import Annotated

PathLike = str | Path


# # # # # CONFIGURE LOGGER # # # # #
logger = logging.getLogger("neuralset")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# # # # # CONFIGURED LOGGER # # # # #


def _int_cast(v: tp.Any) -> tp.Any:
    """casts integers to string"""
    if isinstance(v, int):
        return str(v)
    return v


# type hint for casting integers to string
# this is useful for subject field which can be automatically converted from
# str to int by pandas
StrCast = Annotated[str, pydantic.BeforeValidator(_int_cast)]
CACHE_FOLDER = Path.home() / ".cache/neuralset/"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


class _Module(pydantic.BaseModel):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return []

    @tp.final  # make sure nobody gets it wrong and override it
    def __post_init__(self) -> None:
        """This should not exist in subclasses, as we use pydantic's model_post_init"""

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # get requirements from superclasses as well
        reqs = tuple(x.strip() for x in cls.requirements)
        for base in cls.__bases__:
            breqs = getattr(base, "requirements", ())
            if breqs is not cls.requirements:
                reqs = breqs + reqs
        cls.requirements = reqs

    @classmethod
    def _can_be_instanciated(cls) -> bool:
        return not any(cls.__name__.startswith(k) for k in ["Base", "_"])

    @classmethod
    def install_requirements(cls) -> None:
        cls._check_requirements(install=True)

    @classmethod
    def _check_requirements(cls, install: bool = False) -> None:
        import_names = {
            "pillow": "PIL",
            "scikit-image": "skimage",
            "opencv-python": "cv2",
            "git+https://github.com/nltk/nltk_contrib.git@683961c53f0c122b90fe2d039fe795e0a2b3e997": "nltk_contrib",
        }

        for package in cls.requirements:
            name = package.split(">=")[0]
            name = name.split("==")[0]
            try:
                importlib.import_module(import_names.get(name, name))
            except ModuleNotFoundError:
                if install:
                    # importing pip has border effects on distutils (and make a mess with dino)
                    import pip

                    warnings.warn(f"Installing missing package {name!r} (this may crash)")
                    pip.main(["install", package])
                else:
                    warnings.warn(
                        f"Missing {name!r}. This will "
                        f"likely crash. Use {cls.__name__}"
                        ".install_requirements()"
                    )


class Frequency(float):
    """A float representing a frequency, with extra helpers to
    help convert from seconds to samples and vice-versa
    """

    @tp.overload
    def to_ind(self, seconds: float) -> int: ...

    @tp.overload  # noqa
    def to_ind(self, seconds: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_ind(self, seconds: tp.Any) -> tp.Any:  # noqa
        """Converts a time in seconds (or multiple times in an array)
        to a sample index
        """
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @tp.overload
    def to_sec(self, index: int) -> float: ...

    @tp.overload  # noqa
    def to_sec(self, index: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_sec(self, index: tp.Any) -> tp.Any:  # noqa
        """Converts a sample index to a time in seconds"""
        return index / self

    @staticmethod
    def _yaml_representer(dumper, data):
        "Represents Frequency instances as floats in yamls"
        return dumper.represent_scalar("tag:yaml.org,2002:float", str(float(data)))


yaml.representer.SafeRepresenter.add_representer(Frequency, Frequency._yaml_representer)
