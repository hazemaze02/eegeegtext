# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from math import prod
from pathlib import Path

import pydantic
import pytest
import torch

from neuralset.infra import TaskInfra

from .utils import (
    BaseExperiment,
    StandardScaler,
    WandbLoggerConfig,
    all_subclasses,
    run_grid,
)


class Experiment(BaseExperiment):
    a: int
    b: list[int]
    c: dict[str, tp.Any]
    j: dict[str, dict[str, int]]
    trainer: tp.Any
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def run(self):
        return 1


def test_all_subclasses() -> None:
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(C):
        pass

    out = all_subclasses(A)
    assert out == {B, C, D}


@pytest.mark.parametrize("combinatorial", [True, False])
@pytest.mark.parametrize("dry_run", [True, False])
def test_run_grid(combinatorial: bool, dry_run: bool, tmp_path: Path) -> None:
    folder = str(tmp_path / "test_exp")

    exp = Experiment
    exp_name = "test"
    base_config = {
        "a": 1,
        "b": [2],
        "c": {"d": 3, "e": [4, 5], "f": {"g": "h", "i": 6}},
        "j": {
            "k": {
                "l": 7,
                "m": 8,
            },
        },
        "infra": {
            "folder": folder,
            "job_name": "",
        },
        "trainer": {"wandb_config": {"name": ""}},
    }
    grid: dict[str, list] = {
        "a": [7, 8],
        "b": [[9], [10, 11]],
        "c.d": [12, 13],
        "c.f.g": [14, 15],
        "j.k": [{"l": 16}, {"l": 17}],
    }
    job_name_keys = ["trainer.wandb_config.name"]

    out = run_grid(
        exp,
        exp_name,
        base_config,
        grid,
        job_name_keys=job_name_keys,
        combinatorial=combinatorial,
        overwrite=False,
        dry_run=dry_run,
    )

    func = prod if combinatorial else sum
    assert len(out) == func([len(g) for g in grid.values()])  # type: ignore

    # Make sure overwrite has been covered properly
    if not dry_run:
        run_grid(
            exp,
            exp_name,
            base_config,
            grid,
            job_name_keys,
            combinatorial,
            overwrite=True,
            dry_run=dry_run,
        )


class Model1(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["Model1"] = "Model1"
    param1: int = 0


class Model2(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["Model2"] = "Model2"
    param2: int


class Experiment1(BaseExperiment):
    model: Model1 | Model2 = pydantic.Field(..., discriminator="name")
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def run(self):
        return 1


def test_run_grid_discriminated_union(tmp_path: Path) -> None:
    folder = str(tmp_path / "test_exp")

    exp = Experiment1
    exp_name = "test"
    base_config = {
        "model": {
            "name": "Model1",
        },
        "infra": {
            "folder": folder,
            "job_name": "",
        },
    }
    grid: dict[str, list] = {
        "model": [
            {
                "name": "Model1",
                "param1": 1,
            },
            {
                "name": "Model2",
                "param2": 2,
            },
        ]
    }

    out_configs = run_grid(
        exp,
        exp_name,
        base_config,
        grid,
        job_name_keys=None,
        combinatorial=True,
        overwrite=False,
        dry_run=False,
    )
    assert len(out_configs) == 2


@pytest.mark.parametrize("xp_config", [None, {"a": 1}])
def test_wandb_logger_config(tmp_path, xp_config):
    kwargs = {
        "project": "test",
        "group": "test_grid1",
    }
    logger = WandbLoggerConfig(**kwargs).build(tmp_path, xp_config=xp_config)

    from lightning.pytorch.loggers import WandbLogger

    assert isinstance(logger, WandbLogger)


def test_standard_scaler():
    batch_size, n_latent_dims = 8, 16
    X = torch.rand(batch_size, n_latent_dims)
    scaler = StandardScaler(dim=1)
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=0), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled_X.std(dim=0), torch.tensor(1.0), atol=1e-6)


def test_standard_scaler_3d():
    batch_size, n_latent_dims, n_times = 64, 768, 300
    X = torch.rand(batch_size, n_latent_dims, n_times)
    scaler = StandardScaler(dim=1)
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=(0, 2)), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled_X.std(dim=(0, 2)), torch.tensor(1.0), atol=1e-6)


def test_standard_scaler_partial_fit():
    batch_size, n_latent_dims, n_times = 64, 768, 300
    X = torch.rand(batch_size, n_latent_dims, n_times)
    scaler = StandardScaler(dim=1)
    scaler.partial_fit(X[: batch_size // 2])
    scaler.partial_fit(X[batch_size // 2 :])
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert torch.allclose(scaled_X.mean(dim=(0, 2)), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(scaled_X.std(dim=(0, 2)), torch.tensor(1.0), atol=1e-4)


def test_standard_scaler_near_constant():
    batch_size, n_latent_dims = 8, 4
    X = torch.rand(batch_size, n_latent_dims)
    constant_mask = torch.Tensor([True, False, False, True]).bool()
    X[:, constant_mask] = 18.0
    scaler = StandardScaler(dim=1)
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    assert X.shape == scaled_X.shape
    assert (scaler._scale[constant_mask] == torch.tensor(1.0)).all()
    assert torch.allclose(scaled_X[:, constant_mask], torch.tensor(0.0), atol=1e-6)

    assert torch.allclose(scaled_X.mean(dim=0), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        scaled_X[:, ~constant_mask].std(dim=(0)), torch.tensor(1.0), atol=1e-6
    )
