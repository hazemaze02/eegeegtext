# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the experiment.

We suggest the following structure:
- `Data`: configures dataset and features to return DataLoaders
- `Trainer`: creates the deep learning model and exposes a `fit` and `test` methods
- `Experiment`: main class that defines the experiment to run by using `Data` and `Trainer`
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

import neuralset as ns
from neuralset.infra.task import TaskInfra
from neuraltrain.losses import LossConfig
from neuraltrain.metrics import MetricConfig
from neuraltrain.models import ModelConfig
from neuraltrain.optimizers import OptimizerConfig
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig

from .pl_module import BrainModule


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and features."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.data.StudyLoader
    neuro: ns.features.FeatureConfig
    feature: ns.features.FeatureConfig
    valid_size: float = 0.2
    test_size: float = 0.2
    valid_seed: int | None = None
    # Dataset
    start: float = -0.5
    duration: float = 1.4
    batch_size: int = 64
    num_workers: int = 0
    seed: int | None = None

    def build(self) -> tuple[dict[str, DataLoader], int]:
        events = self.study.build()

        # Split into train/valid/test sets
        stimulus_events = events[events.type == self.feature.name]
        n_classes = stimulus_events.code.nunique()
        train_inds, test_inds = train_test_split(
            stimulus_events.index,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stimulus_events.trigger,
        )
        train_inds, valid_inds = train_test_split(
            train_inds,
            test_size=self.valid_size,
            random_state=self.valid_seed,
            stratify=stimulus_events.loc[train_inds].trigger,
        )
        events.loc[train_inds, "split"] = "train"
        events.loc[valid_inds, "split"] = "valid"
        events.loc[test_inds, "split"] = "test"

        event_summary = (
            events.reset_index()
            .groupby(["split", "type"])[["index", "subject", "filepath", "code"]]
            .nunique()
        )
        print("Event summary: \n", event_summary)

        self.neuro.prepare(events)
        features = {"input": self.neuro, "target": self.feature}

        # Prepare dataloaders
        loaders = {}
        for split in ["train", "valid", "test"]:
            segments = ns.segments.list_segments(
                events,
                idx=events.split == split,
                start=self.start,
                duration=self.duration,
            )
            dataset = ns.SegmentDataset(features=features, segments=segments)
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=self.batch_size,
                shuffle=split == "train",
                num_workers=self.num_workers,
            )

        return loaders, n_classes


class Experiment(BaseExperiment):
    """Defines the main experiment pipeline including data loading and training/evaluation."""

    data: Data
    # Reproducibility
    seed: int = 33
    # Model
    brain_model_config: ModelConfig
    load_checkpoint: bool = True
    # Loss
    loss: LossConfig
    # Optimization
    optim: OptimizerConfig
    # Metrics
    metrics: list[MetricConfig]
    # Weights & Biases
    wandb_config: WandbLoggerConfig | None = None
    # Hardware
    strategy: str = "auto"
    accelerator: str = "gpu"
    # Optim
    n_epochs: int = 10
    patience: int = 5
    limit_train_batches: int | None = None
    # Others
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    # Eval
    checkpoint_path: str | None = None
    test_only: bool = False

    # Internal properties
    _trainer: pl.Trainer | None = None
    _brain_module: BrainModule | None = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        # Update Trainer parameters based on infra
        self.data.num_workers = self.infra.cpus_per_task

    def _init_module(self, model: nn.Module) -> pl.LightningModule:
        # Setup torch-lightning module
        if self.checkpoint_path:
            assert Path(
                self.checkpoint_path
            ).exists(), f"Checkpoint path {self.checkpoint_path} does not exist."
            checkpoint_path = Path(self.checkpoint_path)
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
        if checkpoint_path.exists() and self.load_checkpoint:
            print(f"\nLoading model from {checkpoint_path}\n")  # XXX Use logger
            init_fn = BrainModule.load_from_checkpoint
        else:
            init_fn = BrainModule
            checkpoint_path = None

        pl_module = init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics={metric.log_name: metric.build() for metric in self.metrics},
            max_epochs=self.n_epochs,
            checkpoint_path=checkpoint_path,
        )

        return pl_module

    def _setup_wandb_logger(self) -> WandbLogger | bool:
        if not self.wandb_config:
            wandb_logger = False
        else:
            if self.wandb_config.offline:
                login_kwargs = {
                    "key": "X"
                    * 40,  # https://github.com/wandb/wandb/issues/960#issuecomment-612149459
                }
            else:
                login_kwargs = {
                    "host": "https://fairwandb.org",
                }
            wandb.login(**login_kwargs)
            wandb_logger = self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
            )
            wandb_logger.experiment.config["_dummy"] = None  # To launch initialization
        return wandb_logger

    def _setup_trainer(self, wandb_logger: WandbLogger | None) -> pl.Trainer:
        trainer = pl.Trainer(
            strategy=self.strategy,
            devices=self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor="val_loss",
                    save_on_train_epoch_end=True,
                ),
            ],
            logger=wandb_logger,
        )
        return trainer

    def fit(
        self, train_loader: DataLoader, valid_loader: DataLoader, n_classes: int
    ) -> None:
        # Initialize brain model
        batch = next(iter(train_loader))
        n_in_channels = batch.data["input"].shape[1]
        brain_model = self.brain_model_config.build(
            n_in_channels=n_in_channels, n_outputs=n_classes
        )
        self._brain_module = self._init_module(brain_model)

        wandb_logger = self._setup_wandb_logger()
        self._trainer = self._setup_trainer(wandb_logger)

        self._trainer.fit(
            model=self._brain_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._brain_module.checkpoint_path,
        )

    def test(self, test_loader: DataLoader) -> None:
        self._trainer.test(self._brain_module, dataloaders=test_loader)

    @infra.apply
    def run(self):
        pl.seed_everything(self.seed, workers=True)

        loaders, n_classes = self.data.build()

        if not self.test_only:
            self.fit(loaders["train"], loaders["valid"], n_classes=n_classes)
        self.test(loaders["test"])

        return self._trainer
