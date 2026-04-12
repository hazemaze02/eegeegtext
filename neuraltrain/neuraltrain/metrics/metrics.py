# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation metrics.
"""

# pylint: disable=attribute-defined-outside-init

import typing as tp
from collections import defaultdict

import numpy as np
import torch
import torchmetrics
from torch import nn


class Rank(torchmetrics.Metric):
    """Rank of predictions based on a retrieval set, using cosine similarity.

    Parameters
    ----------
    reduction :
        How to reduce the example-wise ranks.
    max_samples :
        Maximum expected number of instances in the retrieval set. Used to build the internal
        histogram of seen ranks.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(
        self,
        reduction: tp.Literal["mean", "median", "std"] = "median",
        relative: bool = False,
    ):
        super().__init__()

        self.reduction = reduction
        self.relative = relative
        self.add_state(
            "ranks",
            default=torch.Tensor([]),
            dist_reduce_fx="cat",
        )
        self.rank_count: torch.Tensor  # For mypy

    @classmethod
    def _compute_sim(cls, x, y, norm_kind="y", eps=1e-15):
        if norm_kind is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm_kind == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm_kind == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm_kind == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm_kind}.")

        # Normalize inside einsum to avoid creating a copy of candidates which can be pretty big
        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    def _compute_ranks(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> torch.Tensor:
        scores = self._compute_sim(x, y)

        if x_labels is not None and y_labels is not None:
            # Use explicit mapping to match predictions and targets
            true_inds = torch.tensor(
                [y_labels.index(x) for x in x_labels],
                dtype=torch.long,
                device=scores.device,
            )[:, None]
            true_scores = torch.take_along_dim(scores, true_inds, dim=1)
        else:
            # Assume 1:1 mapping of predictions and targets
            assert x_labels is None and y_labels is None
            assert x.shape[0] == y.shape[0]
            true_scores = torch.diag(scores)[:, None]

        # Average ranks obtained with stricly greater-than and greater-than-or-equals operations to
        # account for repeated scores.
        # E.g., the zero-based rank of prediction "1" in [0, 1, 1, 1, 2] will be 2 (instead of 1 or
        # 3).
        ranks_gt = (scores > true_scores).nansum(axis=1)
        ranks_ge = (scores >= true_scores).nansum(axis=1) - 1
        ranks = (ranks_gt + ranks_ge) / 2
        ranks[ranks < 0] = len(scores) // 2  # FIXME

        if self.relative:
            ranks /= len(y)

        return ranks

    @torch.inference_mode()
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> None:
        """Update internal list of ranks.

        Parameters
        ----------
        x :
            Tensor of predictions, of shape (N, F).
        y :
            Tensor of retrieval set examples, of shape (M, F).
        x_labels, y_labels :
            If provided, used to match predictions and ground truths that don't have the same
            number of examples. Should have length of N and M, respectively
        """
        ranks = self._compute_ranks(x, y, x_labels, y_labels)
        self.ranks = torch.cat([self.ranks, ranks])  # type: ignore

    def compute(self) -> torch.Tensor:
        agg_func: tp.Callable
        if self.reduction == "mean":
            agg_func = torch.mean
        elif self.reduction == "median":
            agg_func = torch.median
        elif self.reduction == "std":
            agg_func = torch.std
        else:
            raise ValueError(
                f'Unknown aggregation {self.reduction} for computing metric. Available aggregations are: "mean", "median" or "std".'
            )
        return agg_func(self.ranks)

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str], subjects: None | list[str] = None
    ) -> tp.Dict[str, float]:
        """
        Compute the average rank for each class.
        """
        assert len(ranks) == len(labels)
        groups = defaultdict(list)
        agg_func = np.mean if self.reduction == "mean" else np.median
        if subjects is None:
            for i, label in enumerate(labels):
                groups[label].append(ranks[i])
            return {label: agg_func(ranks) for label, ranks in groups.items()}
        else:
            assert len(subjects) == len(labels)
            for i, label in enumerate(labels):
                groups[label, subjects[i]].append(ranks[i])
            tmp = {
                (label, subject): agg_func(ranks)
                for (label, subject), ranks in groups.items()
            }
            return [
                {label: tmp[label, subject] for label, subject in tmp if subject == s}
                for s in np.unique(subjects)
            ]

    @classmethod
    def _compute_topk_scores(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        y_labels: list[str],
        k: int = 5,
    ) -> tp.Tuple[list[list[str]], list[list[float]]]:
        """
        Compute the top-k predictions and scores for each example in x.
        """
        scores = cls._compute_sim(x, y)
        topk_inds = torch.argsort(scores, dim=1, descending=True)[:, :k]
        topk_labels = [[y_labels[ind] for ind in inds] for inds in topk_inds]
        scores = [
            [scores[i, ind].item() for ind in inds] for i, inds in enumerate(topk_inds)
        ]
        return topk_labels, scores


class TopkAcc(Rank):
    """Top-k accuracy.

    Parameters
    ----------
    topk :
        K in top-k, i.e. minimal rank to classify a prediction as a success.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, topk: int = 5):
        super().__init__(relative=False)
        self.topk = topk

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str], subjects: None | list[str] = None
    ) -> tp.Dict[str, float]:
        """
        Compute the top-k accuracy for each class.
        """
        groups = defaultdict(list)
        if subjects is None:
            for i, label in enumerate(labels):
                groups[label].append(ranks[i])
            return {
                label: np.mean([r < self.topk for r in ranks])
                for label, ranks in groups.items()
            }
        else:
            assert len(subjects) == len(labels)
            for i, label in enumerate(labels):
                groups[label, subjects[i]].append(ranks[i])
            tmp = {
                (label, subject): np.mean([ranks < self.topk for ranks in ranks])
                for (label, subject), ranks in groups.items()
            }
            return [
                {label: tmp[label, subject] for label, subject in tmp if subject == s}
                for s in np.unique(subjects)
            ]

    def compute(self) -> torch.Tensor:
        ranks = self.ranks
        return (ranks < self.topk).float().mean()


class ExpectedAccuracy(torchmetrics.Metric):
    """Expected accuracy, i.e., average of probabilities for the correct class/item.

    Parameters
    ----------
    loss :
        Loss function used during training.

    Notes
    -----
    You have to supply the loss function to compute these probabilities, e.g.:
    loss = ClipLoss()  # used for model training
    ...
    metric = ExpectedAccuracy(loss)
    This is necessary since loss may have learned components like temperature.

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(
        self,
        loss: nn.Module | None = None,
    ):
        super().__init__()

        self.add_state(
            "probs",
            default=torch.Tensor([]),
            dist_reduce_fx="cat",
        )
        self.loss = loss
        self.probs: torch.Tensor  # For mypy

    def _compute_probs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> torch.Tensor:
        assert self.loss is not None
        probs = self.loss.get_probabilities(x, y)  # type: ignore
        if x_labels is not None and y_labels is not None:
            # Use explicit mapping to match predictions and targets
            true_inds = torch.tensor(
                [y_labels.index(x) for x in x_labels],
                dtype=torch.long,
                device=probs.device,
            )[:, None]
            correct_probs = torch.take_along_dim(probs, true_inds, dim=1).squeeze(1)
        else:
            # Assume 1:1 mapping of predictions and targets
            assert x_labels is None and y_labels is None
            assert x.shape[0] == y.shape[0]
            correct_probs = torch.diag(probs)
        return correct_probs

    @torch.inference_mode()
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> None:
        """Update internal list of ranks.

        Parameters
        ----------
        x :
            Tensor of predictions, of shape (N, F).
        y :
            Tensor of retrieval set examples, of shape (M, F).
        x_labels, y_labels :
            If provided, used to match predictions and ground truths that don't have the same
            number of examples. Should have length of N and M, respectively
        """
        probs = self._compute_probs(x, y, x_labels, y_labels)
        self.probs = torch.cat([self.probs, probs])

    def compute(self) -> torch.Tensor:
        return torch.mean(self.probs)
