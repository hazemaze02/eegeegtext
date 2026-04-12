# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pydantic
import torch
import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import neuralset as ns
from neuralset.events import Event


class NeuroLoader(pydantic.BaseModel):
    """
    A flexible pipeline for loading neuro data for a given experiment.

    Parameters
    ----------
    study: ns.data.StudyLoader
        The config of the study: an instance of ns.data.StudyLoader.
    neuro: ns.features.FeatureConfig, optional
        The neuro feature: an instance of ns.features.Meg, ns.features.Eeg or ns.features.Fmri.
        By default, select type automatically based on the study.
    event_type: str
        The type of event to use.
        By default, decode words.
    start: float
        The start of the segments relative to the word onsets.
        Default: -0.5.
    duration: float
        The duration of the segments.
        Default: 3.0.
    """

    study: ns.data.StudyLoader
    neuro: ns.features.FeatureConfig | tp.Literal["auto"] = "auto"
    event_type: str = "Word"
    start: float = -0.5
    duration: float = 3.0

    _neuro_type: str = pydantic.PrivateAttr()

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.event_type != "auto" and self.event_type not in Event._CLASSES:
            raise ValueError(f"Event type {self.event_type} not found in events")

    def load_events(self) -> pd.DataFrame:
        """
        Load the events.
        """
        events = self.study.build()

        assert (
            self.event_type in events.type.unique()
        ), f"Event type {self.event_type} not found in events"
        neuro_types = list(set(events.type) & {"Eeg", "Meg", "Fmri"})
        assert len(neuro_types) == 1, "There are multiple neuro types"
        self._neuro_type = neuro_types[0]

        if self.neuro == "auto":
            infra_config = {"folder": self.study.cache, "keep_in_ram": True}
            if self._neuro_type == "Fmri":
                self.neuro = ns.features.Fmri(
                    detrend=True,
                    mesh="fsaverage5",
                    infra=infra_config,  # type: ignore
                )
            else:
                self.neuro = getattr(ns.features, self._neuro_type)(
                    frequency=50.0,
                    filter=(0.1, 20.0),
                    clamp=20.0,
                    scaler="RobustScaler",
                    infra=infra_config,
                )
        return events

    def load_neuro(self) -> torch.Tensor:
        """
        Load the neuro data.
        """
        events = self.load_events()
        self.neuro.prepare(events)  # type: ignore

        segments = ns.segments.list_segments(
            events,
            idx=events.type == self.event_type,
            start=self.start,
            duration=self.duration,
        )
        dataset = ns.SegmentDataset(
            features={"neuro": self.neuro},  # type: ignore
            segments=segments,
            remove_incomplete_segments=True,
        )
        return dataset.as_one_batch().data["neuro"]

    def get_evoked(self):

        events = self.load_events()
        batch = self.load_neuro()
        assert self._neuro_type in [
            "Eeg",
            "Meg",
        ], "Evoked response only works for EEG/MEG data"

        row = events.query(f'type=="{self._neuro_type}"').iloc[0]
        raw = ns.events.BaseDataEvent.from_dict(row).read()
        raw = raw.pick(self.neuro.pick_types, verbose=False)

        with raw.info._unlock():
            raw.info["sfreq"] = self.neuro.frequency
        epochs = mne.EpochsArray(
            batch,
            info=raw.info,
            events=None,
            verbose=False,
            tmin=self.start,
        )
        evoked = epochs.average(method="median")
        return evoked

    def plot_evoked(self, mne_kwargs: tp.Dict[str, tp.Any] = {}):
        evoked = self.get_evoked()
        fig = evoked.plot_joint(**mne_kwargs)
        return fig


class TimeDecoding(NeuroLoader):
    """
    A flexible pipeline for decoding a target feature from neuro data.

    Parameters
    ----------
    target: ns.features.FeatureConfig, optional
        The target feature.
        By default, select based on the event type.
    model: str, optional
        Whether to use a RidgeCV (regression) or a RidgeClassifierCV (classification).
    """

    target: ns.features.FeatureConfig | tp.Literal["auto"] = "auto"
    model: tp.Literal["RidgeCV", "RidgeClassifierCV"] = "RidgeCV"

    infra: ns.infra.TaskInfra = ns.infra.TaskInfra()
    model_config = pydantic.ConfigDict(extra="forbid")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.event_type != "auto" and self.event_type not in Event._CLASSES:
            raise ValueError(f"Event type {self.event_type} not found in events")

    @infra.apply
    def decode(self) -> np.ndarray:
        """
        Decode the target feature from the neuro data using a simple model (Ridge or logistic regression).
        Returns the decoding scores, of shape (n_subjects, n_times).
        """

        events = self.load_events()

        if self.target == "auto":
            match self.event_type:
                case "Word":
                    languages = events[events.type == "Word"].language.unique()
                    assert len(languages) == 1, "There are multiple languages"
                    language = languages[0]
                    if not isinstance(language, str):
                        language = "english"
                    target = ns.features.WordFrequency(
                        language=language, aggregation="trigger"
                    )
                case "Image":
                    target = ns.features.Image()
                case _:
                    raise NotImplementedError(
                        f"Event type {self.event_type} not implemented in auto mode"
                    )

        self.neuro.prepare(events)  # type: ignore
        features = {"neuro": self.neuro, "target": target}

        scores = []
        for _, subject_events in events.groupby("subject"):
            segments = ns.segments.list_segments(
                subject_events,
                idx=subject_events.type == self.event_type,
                start=self.start,
                duration=self.duration,
            )
            dataset = ns.SegmentDataset(
                features=features,  # type: ignore
                segments=segments,
                remove_incomplete_segments=True,
            ).as_one_batch()
            score = self.get_score(dataset)
            scores.append(score)

        return np.stack(scores)

    def plot_decoding(self):
        scores = self.decode().mean(axis=0)
        t = np.linspace(self.start, self.duration, len(scores))
        fig = plt.figure(figsize=(8, 5))

        plt.plot(t, scores)
        plt.axvline(0, color="black", linestyle="--")
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Pearson correlation" if self.model == "RidgeCV" else "Accuracy (%)")

        return fig

    @staticmethod
    def _pearsonr_corr(x: np.ndarray, y: np.ndarray) -> float:
        return pearsonr(x, y)[0]  # Return correlation only

    def get_score(self, dataset: ns.dataloader.SegmentData) -> np.ndarray:
        X, y = dataset.data["neuro"], dataset.data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        _, _, n_times = X_train.shape
        if y_train.ndim == 1:
            y_train = y_train[:, None]
            y_test = y_test[:, None]
        _, n_features = y_train.shape

        alphas = np.logspace(-2, 8, 7)
        if self.model == "RidgeCV":
            model = RidgeCV(alphas=alphas)
            metric = self._pearsonr_corr
        elif self.model == "RidgeClassifierCV":
            model = RidgeClassifierCV(alphas=alphas)
            metric = accuracy_score
        else:
            raise NotImplementedError()

        scores = np.zeros((n_times, n_features))
        for t in tqdm.trange(X_train.shape[-1], desc="Decoding"):
            model.fit(X_train[:, :, t], y_train)
            y_pred = model.predict(X_test[:, :, t])
            if y_pred.ndim == 1:
                y_pred = y_pred[:, None]
            for d in range(n_features):
                scores[t, d] = metric(y_test[:, d], y_pred[:, d])

        return scores.mean(axis=1)
