# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quick test run on reduced data and number of epochs for CI.
"""

from neuraltrain.utils import update_config

import neuralset as ns

from ..main import Experiment  # type: ignore
from .defaults import default_config  # type: ignore

update = {
    "infra.cluster": None,
    # use same folder as in neuralset tests to share cache:
    "data.study.path": ns.CACHE_FOLDER,
    "data.study.n_timelines": 1,
    "accelerator": "cpu",
    "fast_dev_run": True,
    "wandb_config": None,
}


def test_run(config: dict) -> None:
    task = Experiment(**config)
    task.infra.clear_job()
    trainer = task.run()
    metrics = trainer.logged_metrics.values()
    assert all(isinstance(metric.item(), float) for metric in metrics)


if __name__ == "__main__":
    updated_config = update_config(default_config, update)
    test_run(updated_config)
