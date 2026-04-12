# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid over different configurations of stimulus classification experiment.
"""

from neuraltrain.utils import run_grid, update_config

from ..main import Experiment  # type: ignore
from .defaults import PROJECT_NAME, SAVEDIR, default_config  # type: ignore

GRID_NAME = "hp_search"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": "learnfair",
        "timeout_min": 60,
        "gpus_per_node": 1,
        "cpus_per_task": 10,  # Also used for num_workers
        "job_name": PROJECT_NAME,
    },
    "patience": 15,
    "wandb_config.group": GRID_NAME,
}

grid = {
    "brain_model_config.depth": [2, 4, 16],
    "n_epochs": [100, 0],  # n_epochs=0 for chance-level
    "optim.lr": [3e-4, 1e-3],
    "seed": [33, 87],
}


if __name__ == "__main__":
    updated_config = update_config(default_config, update)

    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["wandb_config.name"],
        combinatorial=True,
        overwrite=True,
        dry_run=False,
        infra_mode="force",
    )
