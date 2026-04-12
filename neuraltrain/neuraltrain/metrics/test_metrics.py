# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from ..losses import ClipLoss
from .metrics import ExpectedAccuracy, Rank, TopkAcc


@pytest.mark.parametrize("reduction", ["median", "mean", "std"])
def test_rank(reduction):
    F = 8

    # Create y_pred such that item i has rank i in y_true
    template = torch.rand(1, F)
    y_pred = template.repeat(F, 1)
    y_true = torch.triu(y_pred)
    y_true_labels = torch.arange(y_true.shape[0]).tolist()
    y_pred_labels = torch.arange(y_pred.shape[0]).tolist()

    metric = Rank(reduction)
    out1 = metric(y_pred, y_true, None, None)
    metric.reset()

    # With labels
    out2 = metric(y_pred, y_true, y_pred_labels, y_true_labels)
    metric.reset()

    # With labels and different shapes
    y_true_repeat = torch.cat([y_true, torch.zeros(F, F)])
    y_true_labels_repeat = torch.arange(y_true_repeat.shape[0]).tolist()
    out3 = metric(y_pred, y_true_repeat, y_pred_labels, y_true_labels_repeat)
    metric.reset()

    # With multiple calls to update
    for _ in range(3):
        out4 = metric(y_pred, y_true, None, None)

    true_reduced_rank = {"mean": torch.mean, "median": torch.median, "std": torch.std}[
        reduction
    ](torch.Tensor(y_pred_labels))
    assert out1 == out2 == out3 == out4 == true_reduced_rank

    # Without labels but with different shapes
    with pytest.raises(AssertionError):
        metric(y_pred[: F // 2], y_true, None, None)

    # Without labels but with more y_pred than y_true
    with pytest.raises(AssertionError):
        metric(y_pred, y_true[: F // 2], None, None)


def test_rank_half_bin():
    F = 4
    y_pred = torch.rand(1, F)
    y_true = torch.rand(F, F)
    y_true[:2, :] = y_pred
    y_pred_labels, y_true_labels = [0], list(range(F))

    metric = Rank("median")
    out = metric(y_pred, y_true, y_pred_labels, y_true_labels)

    assert out == 0.5


@pytest.mark.parametrize("topk", [1, 3, 5])
def test_topk_acc(topk):
    # Create y_pred such that item i has rank i in y_true
    F = 8
    template = torch.rand(1, F)
    y_pred = template.repeat(F, 1)
    y_true = torch.triu(y_pred)
    y_true_labels = torch.arange(y_true.shape[0]).tolist()
    y_pred_labels = torch.arange(y_pred.shape[0]).tolist()

    metric = TopkAcc(topk)
    out = metric(y_pred, y_true, y_pred_labels, y_true_labels)

    # Get true top-k accuracy
    true_ranks = torch.Tensor(
        y_pred_labels
    )  # Because of the way y_pred and y_true were defined
    true_acc = (true_ranks < topk).float().mean()

    assert out == true_acc


def test_expected_accuracy():
    F = 8

    # Create y_pred such that item i has rank i in y_true
    template = torch.rand(1, F)
    y_pred = template.repeat(F, 1)
    y_true = torch.triu(y_pred)
    y_true_labels = torch.arange(y_true.shape[0]).tolist()
    y_pred_labels = torch.arange(y_pred.shape[0]).tolist()
    loss = ClipLoss(
        norm_kind="y",
        temperature=False,
        symmetric=False,
    )
    metric = ExpectedAccuracy()
    metric.loss = loss
    out1 = metric(y_pred, y_true, None, None)
    metric.reset()

    # With labels
    out2 = metric(y_pred, y_true, y_pred_labels, y_true_labels)
    metric.reset()

    # With labels and different shapes
    y_true_repeat = torch.cat([y_true, torch.zeros(F, F)])
    y_true_labels_repeat = torch.arange(y_true_repeat.shape[0]).tolist()
    _ = metric(y_pred, y_true_repeat, y_pred_labels, y_true_labels_repeat)
    metric.reset()

    # With multiple calls to update
    for _ in range(3):
        out4 = metric(y_pred, y_true, None, None)

    assert out1 == out2 == out4
    # Without labels but with different shapes
    with pytest.raises(AssertionError):
        metric(y_pred[: F // 2], y_true, None, None)

    # Without labels but with more y_pred than y_true
    with pytest.raises(AssertionError):
        metric(y_pred, y_true[: F // 2], None, None)
