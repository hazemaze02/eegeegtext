# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration dictionary for project template experiment on MNE sample dataset.
"""
from pathlib import Path

import neuralset as ns

PROJECT_NAME = "mne_sample_clf"
CACHEDIR = f"{ns.CACHE_FOLDER}/cache/{PROJECT_NAME}"
SAVEDIR = f"{ns.CACHE_FOLDER}/results/{PROJECT_NAME}"
DATADIR = f"{ns.CACHE_FOLDER}/data/mnesample2013"
for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

default_config = {
    "infra": {
        "cluster": None,  # Run example locally
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
    },
    "data": {
        "study": {
            "name": "MneSample2013",
            "path": DATADIR,
            "query": None,
            "infra": {"folder": CACHEDIR, "mode": "cached"},
        },
        "neuro": {
            "name": "Meg",
            "frequency": 120.0,
            "filter": (0.5, 25.0),
            "baseline": (0.0, 0.1),
            "scaler": "RobustScaler",
            "clamp": 16.0,
            "infra": {  # Used for loading and preparing raw data
                "keep_in_ram": True,
                "folder": CACHEDIR,
                "cluster": None,
            },
        },
        "feature": {"name": "Stimulus"},
        "valid_size": 0.2,
        "valid_seed": 87,
        "start": -0.1,
        "duration": 0.5,
        "batch_size": 16,
        "num_workers": 0,
    },
    "wandb_config": {
        "log_model": False,
        "project": PROJECT_NAME,
        "group": "default",
    },
    "brain_model_config": {
        "name": "EEGNet",
        "depth": 2,
        "dropout": 0.5,
    },
    "metrics": [
        {
            "log_name": "acc",
            "name": "Accuracy",
            "kwargs": {"task": "multiclass", "num_classes": 4},  # XXX Depends on data
        },
    ],
    "loss": {"name": "CrossEntropyLoss"},
    "optim": {"name": "Adam", "lr": 1e-3},
    "load_checkpoint": False,
    "n_epochs": 20,
    "limit_train_batches": None,
    "patience": 5,
    "strategy": "auto",
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "fast_dev_run": False,
    "seed": 33,
}


if __name__ == "__main__":
    # The following can be used for local debugging/quick tests.

    from ..main import Experiment

    exp = Experiment(
        **default_config,
    )

    exp.infra.clear_job()
    out = exp.run()
    print(out)
