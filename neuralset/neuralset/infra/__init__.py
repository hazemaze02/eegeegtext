# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import os

try:
    os.environ.setdefault("CONFDICT_UID_VERSION", "1")  # default to version 1
    from exca import ConfDict as ConfDict
    from exca import MapInfra as MapInfra
    from exca import TaskInfra as TaskInfra
    from exca import helpers as helpers
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Please run `pip install exca` asap")
    from . import helpers as helpers  # type: ignore
    from .confdict import ConfDict as ConfDict  # type: ignore
    from .map import MapInfra as MapInfra  # type: ignore
    from .task import TaskInfra as TaskInfra  # type: ignore
