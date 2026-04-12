# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility scripts.
"""

from __future__ import annotations

import copy
import shutil
import typing as tp
from itertools import product
from pathlib import Path

import pydantic
import torch

from neuralset.infra import ConfDict, TaskInfra  # type: ignore[import]


def all_subclasses(cls):
    """Get all subclasses of cls recursively."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


class BaseExperiment(pydantic.BaseModel):
    """Base experiment class which require an infra and a 'run' method."""

    infra: TaskInfra = TaskInfra()

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return []

    def run(self):
        raise NotImplementedError


def update_config(config: dict, update: dict) -> dict:
    """
    This enables to modify a nested dictionary, either by updating an individual entry, or by overwriting full sub-dictionaries.
    Note that the behavior is different from that of ConfDict.update, which only updates individual entries (see examples below).

    Parameters
    ----------
    config : dict
        The dictionary to update.
    update : dict
        The dictionary containing the updates.

    Returns
    -------
    dict: The updated dictionary.


    Example
    -------
        config  = {"a": 1, "b": {"c": 2, "d": 3}}
        update1 = {"b.c": 4}
        update2 = {"b": {"e": 5}}
        With update_config, the following holds:
        update_config(config, update1) -> {"a": 1, "b": {"c": 4, "d": 3}}
        update_config(config, update2) -> {"a": 1, "b": {"e": 5}}
        With ConfDict.update, the following holds:
        ConfDict(config).update(update2) -> {"a": 1, "b": {"c": 2, "d": 3, "e": 5}}
    """
    new_config = copy.deepcopy(config)
    for k, v in update.items():
        path = k.split(".")
        c = new_config
        for p in path[:-1]:
            c = c[p]
        c[path[-1]] = v
    return new_config


def run_grid(
    exp_cls: tp.Type[BaseExperiment],
    exp_name: str,
    base_config: dict[str, tp.Any],
    grid: dict[str, list],
    job_name_keys: list[str] | None = None,
    combinatorial: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    infra_mode: str = "retry",
) -> list[ConfDict]:
    """Run grid over provided experiment.

    Parameters
    ----------
    exp_cls :
        Experiment class to instantiate with `grid`. Must have an `infra` attribute, which will be
        updated when instantiating the different experiments of the grid.
    exp_name :
        Name of the base experiment to run.
    grid :
        Dictionary containing values to perform the sweep on.
    base_config :
        Base configuration to update.
    job_name_keys :
       Flattened config key(s) to update with the experiment-specific 'job_name' variable. E.g.,
       can be used to pass the job name to a wandb logger.
    combinatorial :
        If True, run grid over all possible combinations of the grid. If False, run each parameter
        change individually.
    overwrite :
        If True, delete existing experiment-specific folder.
    dry_run :
        If True, do not add tasks to the infra.
    infra_mode :
        Whether to rerun existing or failed experiments.
        - cached: cache is returned if available (error or not),
                otherwise computed (and cached)
        - retry: cache is returned if available except if it's an error,
                otherwise (re)computed (and cached)
        - force: cache is ignored, and result is (re)computed (and cached)

    Returns
    -------
    list :
        List of config dictionaries used for each experiment of the grid.
    """
    # Update savedir of experiment infra
    base_config = base_config
    base_folder = Path(base_config["infra"]["folder"])

    task: BaseExperiment = exp_cls(
        **base_config,
    )

    if combinatorial:
        grid_product = list(dict(zip(grid.keys(), v)) for v in product(*grid.values()))
    else:
        grid_product = [
            {param: value} for param, values in grid.items() for value in values
        ]

    print(f"Launching {len(grid_product)} tasks")

    out_configs = []
    tmp = task.infra.clone_obj(**{"infra.mode": infra_mode})
    with tmp.infra.job_array() as tasks:
        for params in grid_product:
            job_name = ConfDict(params).to_uid()

            config = update_config(base_config, params)
            config = ConfDict(config)  # flatten the config

            folder = base_folder / exp_name / job_name
            if folder.exists():  # FIXME: adapt to checkpointing
                print(f"{folder} already exists.")
                if overwrite and not dry_run:
                    print(f"Deleting {folder}.")
                    shutil.rmtree(folder)
                    folder.mkdir()

            # Update infra and logger
            config["infra.folder"] = str(folder)
            if job_name_keys is not None:
                for key in job_name_keys:
                    config.update({key: str(job_name)})

            if not dry_run:
                task_ = exp_cls(**config)
                tasks.append(task_)

            out_configs.append(config)

    print("Done.")

    return out_configs


class WandbLoggerConfig(pydantic.BaseModel):
    """Pydantic configuration for torch-lightning's wandb logger."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str | None = None
    version: str | None = None
    offline: bool = False
    dir: Path | None = None
    id: str | None = None
    anonymous: bool | None = None
    project: str | None = None
    log_model: str | bool = False
    experiment: tp.Any | None = None
    prefix: str = ""
    checkpoint_name: str | None = None
    entity: str | None = None
    group: str | None = None

    def build(
        self, save_dir: str | Path, xp_config: dict | pydantic.BaseModel | None = None
    ) -> tp.Any:  # tp.Any to avoid lightning import
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(xp_config, pydantic.BaseModel):
            xp_config = xp_config.model_dump()
        return WandbLogger(**self.model_dump(), save_dir=save_dir, config=xp_config)


def _is_constant_feature(
    var: torch.Tensor, mean: torch.Tensor, n_samples: torch.Tensor
) -> torch.Tensor:
    """Detect if a feature is indistinguishable from a constant feature (on torch Tensors).

    See `sklearn.preprocessing._data._is_constant_feature`.
    """
    eps = torch.finfo(torch.float32).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound


class StandardScaler(pydantic.BaseModel):
    """Standard scaler that can be fitted by batch and handles 2-dimensional features."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: tp.Literal["StandardScaler"] = "StandardScaler"
    dim: int = 1  # Dimension across which the statistics should be computed

    # Internal
    _mean: torch.Tensor | None = None
    _var: torch.Tensor | None = None
    _scale: torch.Tensor | None = None
    _original_shape: list | None = None
    _n_samples_seen: int = 0

    def _reset(self):
        self._mean = None
        self._var = None
        self._scale = None
        self._original_shape = None
        self._n_samples_seen = 0

    def _transpose_flatten(self, X: torch.Tensor) -> torch.Tensor:
        """Transpose and flatten to have (n_total_examples, n_latent_dims)."""
        if X.ndim > 2:
            self._original_shape = [s for i, s in enumerate(X.shape) if i != self.dim]
            X = X.transpose(self.dim, -1).flatten(end_dim=-2)
        return X

    def _unflatten_untranspose(self, X: torch.Tensor) -> torch.Tensor:
        if self._original_shape is not None:
            X = X.unflatten(dim=0, sizes=self._original_shape).transpose(self.dim, -1)
        return X

    def partial_fit(self, X: torch.Tensor) -> StandardScaler:
        X = self._transpose_flatten(X)
        m = self._n_samples_seen
        n = X.shape[0]

        # Update mean
        previous_mean = (
            torch.zeros(X.shape[1], device=X.device) if self._mean is None else self._mean
        )
        batch_mean = X.mean(dim=0)
        self._mean = (m / (m + n)) * previous_mean + (n / (m + n)) * batch_mean

        # Update variance
        previous_var = (
            torch.zeros(X.shape[1], device=X.device) if self._var is None else self._var
        )
        self._var = (
            (m / (m + n)) * previous_var
            + (n / (m + n)) * X.var(dim=0)
            + (m * n / (m + n) ** 2) * (previous_mean - batch_mean) ** 2
        )
        scale = self._var.sqrt()  # type: ignore

        # Compute near-constant mask to avoid scaling by 0
        constant_mask = _is_constant_feature(self._var, self._mean, self._n_samples_seen)  # type: ignore
        scale[constant_mask] = 1.0
        self._scale = scale  # type: ignore
        self._n_samples_seen += n

        return self

    def fit(self, X: torch.Tensor) -> StandardScaler:
        self._reset()
        return self.partial_fit(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clone()
        X = self._transpose_flatten(X)
        X = (X - self._mean.to(X.device)) / self._scale.to(X.device)  # type: ignore
        X = self._unflatten_untranspose(X)
        return X
