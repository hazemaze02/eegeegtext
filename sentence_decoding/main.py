# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pydantic
import wandb
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sentence_decoding.callbacks import InitialEvaluation, TestRetrieval
from sentence_decoding.pl_module import BrainModule
from sentence_decoding.utils import (
    LANGUAGES,
    ShuffledSegmentDataset,
    ShuffleSentences,
    StandardScaler,
    preprocess_text,
)
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralset as ns
from neuralset.dataloader import SegmentDataset
from neuralset.infra.task import TaskInfra
from neuralset.splitting import DeterministicSplitter, set_event_split
from neuraltrain.losses import LossConfig
from neuraltrain.metrics import MetricConfig
from neuraltrain.models import ModelConfig

from .decoder import Decoder


class TrainerConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")

    lr: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 100
    patience: int = 5
    monitor: str = "val_loss"
    transformer_start_epoch: int = 0
    masking_ratio: float = 0.0
    fast_dev_run: bool = False
    gradient_clip_val: float = 0.0


class Data(pydantic.BaseModel):
    """No need for infra (already in meg/images)"""

    model_config = pydantic.ConfigDict(extra="forbid")

    cache: str
    dataset: str | tp.List[str] = "Gwilliams2022"
    data_path: str
    n_timelines: int | tp.Literal["all"] = "all"
    n_timelines_per_subject: int | tp.Literal["all"] = "all"
    n_subjects: int | tp.Literal["all"] = "all"
    neuro: ns.features.FeatureConfig
    feature: ns.features.FeatureConfig
    # Dataset
    start: float = -0.5
    duration: float | tp.Tuple[float, float] | None = 3
    batch_size: int = 64
    batch_size_eval: int | None = None
    num_workers: int = 10
    event_type: str = "Word"

    infra: TaskInfra = TaskInfra()

    def model_post_init(self, __context: tp.Any) -> None:
        if self.dataset in ["Broderick2019", "Accou2023", "Nieuwland2018", "TestEeg2024"]:
            self.neuro.pick_types = ("eeg",)
            self.neuro.name = "Eeg"
            self.neuro = ns.features.Eeg(**self.neuro.dict())
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        language = LANGUAGES.get(self.dataset[0])
        if hasattr(self.feature, "language"):
            self.feature.language = language
        super().model_post_init(__context)

    @infra.apply
    def get_events(self) -> pd.DataFrame:
        dfs = []
        n_subjects = 0
        for dataset in self.dataset:
            study = ns.data.StudyLoader(
                name=dataset,
                path=self.data_path,
                cache=self.cache,
                download=False,
                install=False,
                n_timelines=self.n_timelines,
                max_workers=self.num_workers,
            )

            events = study.build()
            events["subject"] = events.groupby("subject").ngroup()
            events["subject"] = events["subject"].apply(
                lambda x: str(int(x) + n_subjects)
            )
            n_subjects += len(events.subject.unique())
            if self.n_timelines_per_subject != "all":
                timelines_to_keep = []
                for subject, df in events.groupby("subject"):
                    subject_timelines = df.timeline.unique()[
                        : self.n_timelines_per_subject
                    ]
                    timelines_to_keep.extend(subject_timelines)
                events = events.loc[events.timeline.isin(timelines_to_keep)]
            if self.n_subjects != "all":
                subjects = events.subject.unique()
                np.random.shuffle(subjects)
                subjects_to_keep = subjects[: self.n_subjects]
                events = events.loc[events.subject.isin(subjects_to_keep)]
            dfs.append(events)
        events = pd.concat(dfs, ignore_index=True)

        n_subjects = len(events.subject.unique())
        n_words = len(events.loc[events.type == "Word"].text.unique())
        print(f"Loaded {n_subjects} subjects and {n_words} words")

        is_valid_word = events.text.apply(lambda x: isinstance(x, str))
        neuro_type = self.neuro.event_type.__name__
        events = events.loc[
            (events.type == neuro_type) | (events.type == "Sound") | is_valid_word
        ]

        print("Preprocessing text")

        events = preprocess_text(events)

        if dataset in [
            "PallierRead2023",
            "PallierListen2023",
            "Broderick2022",
            "Nieuwland2018",
        ]:
            split_attribute = "sequence_id"
        else:
            split_attribute = "sentence"
        if dataset == "Nieuwland2018":
            events["sequence_id"] = events.groupby("sentence").ngroup() // 2
        splitter = DeterministicSplitter({"train": 0.8, "val": 0.1, "test": 0.1})
        valid = ~events[split_attribute].isna()
        events.loc[valid, "split"] = (
            events.loc[valid, split_attribute].apply(str).apply(splitter)
        )

        train_sentences, val_sentences = (
            events[events.split == "train"].sentence.dropna().unique(),
            events[events.split == "val"].sentence.dropna().unique(),
        )
        overlap = set(train_sentences) & set(val_sentences)
        overlap_ratio = len(overlap) / len(val_sentences)
        print(f"Train/test overlap ratio: {overlap_ratio:.2f}")
        if overlap_ratio > 0.1:
            raise ValueError(f"Overlap ratio is too high: {overlap_ratio:.2f}")
        # remove the overlapping sentences
        sel = events.sentence.isin(overlap)
        events = events.loc[~sel]

        return events

    def get_loaders(self, events):
        neuro_type = self.neuro.event_type.__name__

        if not isinstance(self.feature, ns.features.audio.BaseAudio):  # text
            self.feature.__class__.event_type = getattr(ns.events, self.event_type)
        else:  # audio
            events = set_event_split(events, "Sound", "Word")

        self.feature.prepare(events)
        self.neuro.prepare(events)
        subject_id = ns.features.LabelEncoder(
            event_types=neuro_type, event_field="subject"
        )
        subject_id.prepare(events)
        subject_id.__class__.event_type = getattr(ns.events, neuro_type)
        if neuro_type in ["Meg", "Eeg"]:
            channel_positions = ns.features.ChannelPositions(meg=self.neuro)
            events = channel_positions.prepare(events)
            extra_neuro_features = dict(channel_positions=channel_positions)
        else:
            extra_neuro_features = dict()

        features = {
            "neuro": self.neuro,
            "feature": self.feature,
            "subject_id": subject_id,
            **extra_neuro_features,
        }

        is_trigger = events.type == self.event_type

        # Prepare dataloaders
        loaders = dict()
        for split in ["train", "val", "test"]:
            if split == "train":
                batch_size = self.batch_size
            else:
                batch_size = self.batch_size_eval or self.batch_size
            kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            if split == "train":
                segments = events.ns.list_segments(
                    is_trigger & (events.split == split),
                    start=self.start,
                    duration=self.duration,
                )
                dataset = ShuffledSegmentDataset(
                    features,
                    segments,
                    remove_incomplete_segments=True,
                )
                loaders[split] = DataLoader(
                    dataset, collate_fn=dataset.collate_fn, **kwargs
                )
            else:
                datasets = []
                for dataset_name in self.dataset:
                    segments = events.ns.list_segments(
                        is_trigger
                        & (events.split == split)
                        & (events.study == dataset_name),
                        start=self.start,
                        duration=self.duration,
                    )
                    dataset = SegmentDataset(
                        features,
                        segments,
                        remove_incomplete_segments=True,
                    )
                    datasets.append(dataset)
                loaders[split] = [
                    DataLoader(dataset, collate_fn=dataset.collate_fn, **kwargs)
                    for dataset in datasets
                ]

        return loaders

    def prepare(self):
        events = self.get_events()
        loaders = self.get_loaders(events)
        return loaders


class Experiment(pydantic.BaseModel):
    """ """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    data: Data
    brain_model_config: ModelConfig
    transformer_config: ModelConfig
    use_transformer: bool = False
    use_target_scaler: bool = False

    loss: LossConfig
    metrics: list[MetricConfig]
    retrieval_metrics: list[MetricConfig]
    lm_path: str | None = None

    use_wandb: bool = True
    save_checkpoints: bool = True
    reload_checkpoint: str | None = None
    cache: str
    project: str
    seed: int = 0

    # Optim
    trainer_config: TrainerConfig

    # Internal properties
    _trainer: pl.Trainer | None = None
    _logger: WandbLogger | None = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def load_module(
        self,
        model: nn.Module,
        transformer: nn.Module,
        best: bool = False,
        target_scaler: StandardScaler = None,
    ):
        if self.reload_checkpoint:
            checkpoint_path = self.reload_checkpoint
        else:
            if best:
                checkpoint_path = os.path.join(self.infra.folder, "best.ckpt")
            else:
                checkpoint_path = os.path.join(self.infra.folder, "last.ckpt")
        if os.path.exists(checkpoint_path):
            print(f"\nLoading model {checkpoint_path}\n")
            init_fn = partial(BrainModule.load_from_checkpoint, strict=False)
        else:
            init_fn = BrainModule
            checkpoint_path = None

        pl_module = init_fn(
            checkpoint_path=checkpoint_path,
            model=model,
            transformer=transformer,
            target_scaler=target_scaler,
            loss=self.loss.build(),
            metrics={metric.log_name: metric.build() for metric in self.metrics},
            retrieval_metrics={
                metric.log_name: metric.build() for metric in self.retrieval_metrics
            },
            trainer_config=self.trainer_config,
        )
        pl_module.checkpoint_path = checkpoint_path

        return pl_module

    def get_model(self, train_loader: DataLoader):
        batch = next(iter(train_loader))
        batch_size, n_in_channels, n_timesteps = batch.data["neuro"].shape
        n_outputs = batch.data["feature"].shape[1]
        print("Neuro shape: ", batch.data["neuro"].shape)
        print("Feature shape: ", batch.data["feature"].shape)
        extra_kwargs = {}

        brain_model = self.brain_model_config.build(
            n_in_channels=n_in_channels, n_outputs=n_outputs, **extra_kwargs
        )
        if self.use_transformer:
            transformer = self.transformer_config.build(
                dim=n_outputs,
            )
        else:
            transformer = None
        return brain_model, transformer

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> None:
        brain_model, transformer = self.get_model(train_loader)

        if self.use_target_scaler:
            target_scaler = StandardScaler(dim=1)
            for batch in tqdm(train_loader, "Fitting target scaler"):
                target_scaler.partial_fit(batch.data["feature"])
                # if target_scaler.n_samples_seen_ > 5e5:
                #     break
        else:
            target_scaler = None

        self._brain_module = self.load_module(
            brain_model, transformer, target_scaler=target_scaler
        )

        if self.use_wandb:
            print(self.project)
            wandb.login(host="https://fairwandb.org")
            exp_group, exp_name = self.infra.folder.split("/")[-2:]
            self._logger = WandbLogger(
                save_dir=self.infra.folder,
                project=self.project,
                group=exp_group,
                name=exp_name,
                config=self.model_dump(),
                resume=True,
            )
        else:
            self._logger = None

        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor=self.trainer_config.monitor,
                patience=self.trainer_config.patience,
                mode="max" if "acc" in self.trainer_config.monitor else "min",
                verbose=True,
            ),
            # RichProgressBar(leave=True),
            ShuffleSentences(),
            InitialEvaluation(),
        ]
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor=self.trainer_config.monitor,
                    mode="max" if "acc" in self.trainer_config.monitor else "min",
                    save_on_train_epoch_end=True,
                )
            )
        if self.retrieval_metrics:
            if self.lm_path and Path(self.lm_path).exists():
                decoder = Decoder(
                    lm_path=lm_path,
                    beam_size=20,
                    max_labels_per_timestep=10,
                    lm_weight=1,
                )
            else:
                decoder = None
            callbacks.append(
                TestRetrieval(
                    event_type=self.data.event_type,
                    retrieval_set_sizes=[None, 250],
                    decoder=decoder,
                )
            )

        self._trainer = pl.Trainer(
            # strategy="auto",
            gradient_clip_val=self.trainer_config.gradient_clip_val,
            devices=self.infra.gpus_per_node,
            limit_train_batches=None,
            max_epochs=self.trainer_config.n_epochs,
            enable_progress_bar=True,
            log_every_n_steps=20,
            fast_dev_run=self.trainer_config.fast_dev_run,
            logger=self._logger,
            callbacks=callbacks,
        )

        pl.seed_everything(self.seed)

        # Train model
        self._trainer.fit(
            model=self._brain_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._brain_module.checkpoint_path,
        )
        return self._brain_module

    def test(self, test_loader: DataLoader):
        self._trainer.test(self._brain_module, dataloaders=test_loader, ckpt_path="best")

    def setup_run(self):
        if self.infra.cluster and self.infra.status() != "not submitted":
            for out_type in ["stdout", "stderr"]:
                old_path = Path(getattr(self.infra.job().paths, out_type))
                new_path = Path(self.infra.folder) / f"log.{out_type}"
                if new_path.exists():
                    os.remove(new_path)
                os.symlink(
                    old_path,
                    new_path,
                )
        config_path = Path(self.infra.folder) / "config.yaml"
        if not config_path.exists():
            os.makedirs(self.infra.folder, exist_ok=True)
            with open(config_path, "w") as outfile:
                yaml.dump(self.model_dump(), outfile, indent=4, default_flow_style=False)

        loaders = self.data.prepare()

        return loaders

    @infra.apply
    def run(self):
        loaders = self.setup_run()

        self.fit(loaders["train"], loaders["val"])

        self.test(loaders["test"])

        return self._trainer
