# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd
import pydantic
import torch
from lightning import Callback, LightningModule, Trainer
from torch import nn

import neuralset
import neuralset.segments
from neuralset.dataloader import SegmentDataset
from neuralset.enhancers import BaseEnhancer

LANGUAGES = {
    "Broderick2019": "english",
    "Gwilliams2022": "english",
    "Armeni2022": "english",
    "PallierListen2023": "french",
    "PallierRead2023": "french",
    "SchoffelenListen2019": "dutch",
    "SchoffelenRead2019": "dutch",
    "Nieuwland2018": "english",
    "Accou2023": "dutch",
}


def shuffle_sentences(
    segments: tp.List[neuralset.segments.Segment],
) -> tp.List[neuralset.segments.Segment]:
    """
    Shuffles the segments by blocks of sentences.
    """
    segment_dict = defaultdict(list)
    for segment in segments:
        key = (segment._trigger["timeline"], segment._trigger["sequence_id"])
        segment_dict[key].append(segment)
    keys = list(segment_dict.keys())
    np.random.shuffle(keys)
    res = [segment for key in keys for segment in segment_dict[key]]
    return res


class ShuffleSentences(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.train_dataloader.dataset.shuffle()
        return super().on_train_epoch_start(trainer, pl_module)


class ShuffledSegmentDataset(SegmentDataset):
    def shuffle(self):
        self.segments = shuffle_sentences(self.segments)


def preprocess_text(events):
    remove_special_chars = lambda s: "".join(
        e for e in s if e.isalnum() or e in ["-", "'"]
    )
    preprocess = lambda s: remove_special_chars(s).lower()

    sel = events.type == "Word"
    events.loc[sel, "text"] = events.loc[sel, "text"].apply(preprocess)
    # events.loc[sel, "sentence"] = events.loc[sel, "sentence"].apply(preprocess)
    # events = events.loc[(events.text != "") & (events.text != " ")]
    return events


class TextPreprocessor(BaseEnhancer):
    name: tp.Literal["TextPreprocessor"] = "TextPreprocessor"

    def __init__(self):
        super().__init__()

    def enhance(self, events):
        return preprocess_text(events)


def agg_per_group(
    x: torch.Tensor,
    groups: list[tp.Any] | None = None,
    min_n_ex_per_group: int | None = None,
    max_n_ex_per_group: int | None = None,
    agg_func: tp.Literal["mean", "median", "first"] = "median",
) -> tp.Tuple[torch.Tensor, list]:
    """Aggregate tensors by groups.

    Parameters
    ----------
    x :
        Tensor of shape (N, *) to aggregate per group.
    groups :
        List of length N of labels for the samples used to group `x`.
    min_n_ex_per_group :
        Minimum number of examples in a group for a group to be kept, i.e. ignore a group if there
        are fewer than this number of examples in it.
    max_n_ex_per_group :
        Maximum number of examples in a group. If there are more than this number of examples in a
        group, only the first `max_n_ex_per_group` examples are kept.
    agg_func :
        Aggregation function to use.

    Returns
    -------
    torch.Tensor :
        Aggregated version of `x`, of shape (M, *).
    list :
        Aggregated version of `groups`, of length M.
    """
    if groups is None or len(groups) == 0:
        return x, []
    else:
        assert len(x) == len(
            groups
        ), f"x and groups must have the same length, found {len(x)} and {len(groups)}"
        # NOTE: Using pandas is faster than computing masks by hand.
        groups_df = pd.DataFrame({"label": groups})
        agg_x, agg_groups = list(), list()
        for name, group in groups_df.groupby("label", sort=False):
            if min_n_ex_per_group is not None and len(group) < min_n_ex_per_group:
                # Ignore group because it has too few examples
                continue

            if agg_func == "first":
                if not (x[group.index] == x[group.index[0]]).all():
                    msg = f'Using agg_func="first" but group {name} contains different values for x.'
                    # raise ValueError(msg)
                    # import warnings

                    # warnings.warn(msg)
                agg_out = x[group.index[0]]
            elif agg_func in ["mean", "median"]:
                assert isinstance(x, torch.Tensor)
                indices = group.index
                if max_n_ex_per_group is not None:
                    indices = indices.to_numpy().copy()
                    np.random.shuffle(indices)
                    indices = indices[:max_n_ex_per_group]
                if agg_func == "mean":
                    agg_out = x[indices].mean(dim=0)
                elif agg_func == "median":
                    agg_out = x[indices].median(dim=0)[0]
            else:
                raise NotImplementedError

            agg_x.append(agg_out)
            agg_groups.append(name)

        agg_x = torch.stack(agg_x)

        return agg_x, agg_groups


def agg_retrieval_preds(
    y_pred: torch.Tensor,
    groups_pred: list,
    subjects_pred: list | None = None,
    keep_subject_group_name: bool = False,
    n_ex_per_group: int | None = None,
) -> tp.Tuple[torch.Tensor, list]:
    """Aggregate predictions at the right level before retrieval (e.g. subject-level).

    Parameters
    ----------
    y_pred :
        Tensor of shape (N, *) containing the predictions.
    groups_pred :
        List of length N containing the labels (group) of each example in `y_pred`.
    subjects_pred :
        If provided, list of length M containing an additional grouping variable for `y_pred`. For
        instance, this can be the subject from which prediction comes from, such that examples can
        be aggregated within each subject.
    keep_subject_group_name :
        If True and `subjects_pred` is provided, return `_groups_pred` as a list of tuples
        (subj, group). If False, return `_groups_pred` as a list of items taken from `groups_pred`.
    n_ex_per_group :
        If provided, only keep the first `n_ex_per_group` examples from each group.

    Returns
    -------
    torch.Tensor :
        Aggregated version of `y_pred` using `groups_pred`, of shape (N', *).
    list :
        List of length N' containing the labels (group) for each predicted output example.
    """

    # Average predictions across instances
    groups = (
        groups_pred
        if subjects_pred is None
        else [(subj, inst) for subj, inst in zip(subjects_pred, groups_pred)]
    )
    _y_pred, _groups_pred = agg_per_group(
        y_pred,
        groups=groups,
        agg_func="mean",
        min_n_ex_per_group=n_ex_per_group,
        max_n_ex_per_group=n_ex_per_group,
    )

    _groups_pred = (
        _groups_pred
        if subjects_pred is None or keep_subject_group_name
        else [group[1] for group in _groups_pred]
    )

    return _y_pred, _groups_pred


class StandardScaler(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: tp.Literal["StandardScaler"] = "StandardScaler"
    dim: int = 1
    # infra: TaskInfra = TaskInfra()

    # Internal
    mean_: torch.Tensor | None = None
    var_: torch.Tensor | None = None
    scale_: torch.Tensor | None = None
    original_shape_: list | None = None
    n_samples_seen_: int = 0

    def _reset(self):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.original_shape_ = None
        self.n_samples_seen_ = 0

    def _transpose_flatten(self, X: torch.Tensor) -> torch.Tensor:
        """Transpose and flatten to have (n_total_examples, n_latent_dims)."""
        if X.ndim > 2:
            self.original_shape_ = [s for i, s in enumerate(X.shape) if i != self.dim]
            X = X.transpose(self.dim, -1).flatten(end_dim=-2)
        return X

    def _unflatten_untranspose(self, X: torch.Tensor) -> torch.Tensor:
        if self.original_shape_ is not None:
            X = X.unflatten(dim=0, sizes=self.original_shape_).transpose(self.dim, -1)
        return X

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> nn.Module:
        X = self._transpose_flatten(X)
        assert X.ndim == 2
        m = self.n_samples_seen_
        n = X.shape[0]

        # Update mean
        previous_mean = (
            torch.zeros(X.shape[1], device=X.device) if self.mean_ is None else self.mean_
        )
        batch_mean = X.mean(dim=0)
        self.mean_ = (m / (m + n)) * previous_mean + (n / (m + n)) * batch_mean

        # Update variance
        previous_var = (
            torch.zeros(X.shape[1], device=X.device) if self.var_ is None else self.var_
        )
        self.var_ = (
            (m / (m + n)) * previous_var
            + (n / (m + n)) * X.var(dim=0)
            + (m * n / (m + n) ** 2) * (previous_mean - batch_mean) ** 2
        )
        self.scale_ = self.var_.sqrt()
        self.n_samples_seen_ += n

        return self

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> nn.Module:
        self._reset()
        return self.partial_fit(X, y)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clone()
        X = self._transpose_flatten(X)
        X = X - self.mean_.to(X.device)  # / self.scale_.to(X.device)
        X = self._unflatten_untranspose(X)
        return X
