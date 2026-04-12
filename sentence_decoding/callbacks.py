# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from neuralset.infra.utils import environment_variables
from neuraltrain.metrics import Rank

from .decoder import Decoder
from .utils import agg_per_group, agg_retrieval_preds


class InitialEvaluation(Callback):
    """
    Run an initial evaluation before training starts to get chance level baseline.
    """

    def __init__(self):
        pass

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.validate_loop.run()
        for metric in pl_module.metrics.values():
            metric.reset()
        return


class TestRetrieval(Callback):
    """Accumulate predictions on entire test set before evaluating a metric."""

    def __init__(
        self,
        event_type="Word",
        event_field="text",
        retrieval_set_sizes=[None, 250],
        decoder: Decoder | None = None,
    ):
        self.event_type = event_type
        self.event_field = event_field
        self.retrieval_set_sizes = retrieval_set_sizes
        self.full_outputs = {}
        self.decoder = decoder

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if not hasattr(pl_module, "retrieval_metrics") and not isinstance(
            pl_module.retrieval_metrics, nn.ModuleDict
        ):
            raise ValueError(
                "The LightningModule needs a retrieval_metrics ModuleDict that contains the "
                "metrics to evaluate on the full test set."
            )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.val_dataloaders))
        }

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.test_dataloaders))
        }

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._collate_outputs(outputs, batch, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._collate_outputs(outputs, batch, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._compute_metrics(trainer, pl_module, step_name="val")
        self._save_outputs(trainer, step_name="val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._compute_metrics(trainer, pl_module, step_name="test")
        self._save_outputs(trainer, step_name="test")

    def _save_outputs(self, trainer, step_name):
        n_loaders = (
            len(trainer.val_dataloaders)
            if step_name == "val"
            else len(trainer.test_dataloaders)
        )
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            for key in ["y_pred", "y_true"]:
                full[key] = torch.cat(full[key], dim=0)

            save_dir = os.path.join(trainer.logger.save_dir, "retrieval_outputs")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(full, os.path.join(save_dir, f"{step_name}_{dataloader_idx}.pt"))

    def _collate_outputs(self, outputs, batch, dataloader_idx):
        y_pred, y_true = outputs
        full = self.full_outputs[dataloader_idx]

        full["y_pred"].append(y_pred.cpu())
        full["y_true"].append(y_true.cpu())

        for segment in batch.segments:
            trigger = segment._trigger
            full[self.event_field].append(trigger[self.event_field])
            full["subject_id"].append(trigger["subject"])
            full["sequence_id"].append(trigger["sequence_id"])
            full["timeline"].append(trigger["timeline"])

    def _compute_metrics(self, trainer, pl_module, step_name):
        n_loaders = (
            len(trainer.val_dataloaders)
            if step_name == "val"
            else len(trainer.test_dataloaders)
        )
        retrieval_metrics = {
            k: v
            for k, v in pl_module.retrieval_metrics.items()
            if (k.startswith(step_name) and "retrieval" in k)
        }
        sentence_metrics = {
            k: v
            for k, v in pl_module.retrieval_metrics.items()
            if (k.startswith(step_name) and "sentence" in k)
        }
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            groups_pred = full[self.event_field]
            subjects_pred = full["subject_id"]
            sentence_pred = full["sequence_id"]
            timeline_pred = full["timeline"]
            sentence_uids = [
                f"{sequence}_{timeline}"
                for sequence, timeline in zip(sentence_pred, timeline_pred)
            ]
            y_pred = torch.cat(full["y_pred"], dim=0)
            y_true = torch.cat(full["y_true"], dim=0)

            for retrieval_set_size in self.retrieval_set_sizes:
                out = self._get_retrieval_metrics(
                    y_pred,
                    y_true,
                    groups_pred,
                    subjects_pred,
                    retrieval_metrics,
                    retrieval_set_size=retrieval_set_size,
                )
                for key, value in out.items():
                    key += f"_{dataloader_idx}"
                    pl_module.log(key, value)

            if step_name == "val":
                # keep only one subject for sentence metrics (quite slow)
                idx = np.where(np.array(subjects_pred) == subjects_pred[0])[0]
                y_pred, y_true = y_pred[idx], y_true[idx]
                groups_pred = [groups_pred[i] for i in idx]
                sentence_uids = [sentence_uids[i] for i in idx]
            out_metrics, true_sentences, pred_sentences, corr_sentences = (
                self._get_sentence_metrics(
                    y_pred,
                    y_true,
                    groups_pred,
                    sentence_uids,
                    sentence_metrics,
                )
            )
            for key, value in out_metrics.items():
                key += f"_{dataloader_idx}"
                pl_module.log(key, value)
            save_dir = os.path.join(
                trainer.logger.save_dir,
                f"decoded_sentences",
            )
            os.makedirs(save_dir, exist_ok=True)

            with open(
                os.path.join(save_dir, f"{step_name}_{dataloader_idx}.txt"), "w"
            ) as f:
                for true, pred, corr in zip(
                    true_sentences, pred_sentences, corr_sentences
                ):
                    f.write(f"True: {true}\n")
                    f.write(f"Pred: {pred}\n")
                    f.write(f"Corr: {corr}\n")
                    f.write("\n")
            f.close()

    def _get_sentence_metrics(
        self,
        y_pred,
        y_true,
        true_words,
        sentence_uids,
        metrics,
    ):

        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=true_words, agg_func="first"
        )
        scores = Rank._compute_sim(y_pred, agg_y_true)

        if self.decoder:
            self.decoder.id2word = {i: w for i, w in enumerate(agg_groups_true)}
        scores = torch.Tensor(scores)

        pred_sentences, true_sentences, accs, corr_sentences = [], [], [], []
        for sentence_uid in np.unique(sentence_uids):
            idx = np.where(np.array(sentence_uids) == sentence_uid)[0]
            sentence_scores = scores[idx]

            true_sentence = " ".join([true_words[i] for i in idx])
            pred_sentence = " ".join(
                [agg_groups_true[i] for i in sentence_scores.argmax(dim=1)]
            )
            if self.decoder is not None:
                corr_sentence = (
                    self.decoder.decode(sentence_scores)
                    if self.decoder
                    else pred_sentence
                )
            else:
                corr_sentence = pred_sentence

            pred_sentences.append(pred_sentence)
            true_sentences.append(true_sentence)
            corr_sentences.append(corr_sentence)

            acc = np.mean(
                [
                    w1 == w2
                    for w1, w2 in zip(pred_sentence.split(" "), true_sentence.split(" "))
                ]
            )
            accs.append(acc)
        # sort by accuracy
        idx = np.argsort(accs)[::-1]
        pred_sentences = [pred_sentences[i] for i in idx]
        true_sentences = [true_sentences[i] for i in idx]
        corr_sentences = [corr_sentences[i] for i in idx]

        out = {}
        for correct in [False, True]:
            for metric_name, metric in metrics.items():
                metric_name += f"_correct={correct}"
                preds = corr_sentences if correct else pred_sentences
                with environment_variables(TOKENIZERS_PARALLELISM="false"):
                    res = metric(preds, true_sentences)
                if "bert" in metric_name:
                    res = torch.mean(res["f1"])
                out[metric_name] = res

        return out, true_sentences, pred_sentences, corr_sentences

    @classmethod
    def _get_retrieval_metrics(
        cls, y_pred, y_true, groups_pred, subjects_pred, metrics, retrieval_set_size=None
    ):
        out = {}

        # Keep only the most frequent groups
        if retrieval_set_size is not None:
            groups_df = pd.DataFrame({"label": groups_pred})
            counts = groups_df.label.value_counts()
            most_frequent = set(counts.index[:retrieval_set_size])
            indices = groups_df.label.isin(most_frequent).values
            indices = np.where(indices)[0]  # Get indices where condition is True
            indices = torch.from_numpy(indices)
            y_pred, y_true = y_pred[indices], y_true[indices]
            groups_pred = [groups_pred[i] for i in indices]
            subjects_pred = [subjects_pred[i] for i in indices]

        # Remove repetitions in retrieval set
        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=groups_pred, agg_func="first"
        )

        for metric_name, metric in metrics.items():
            metric = metric.to("cpu")
            if metric_name.endswith("subject-agg"):
                subjects = subjects_pred
            elif metric_name.endswith("instance-agg"):
                subjects = None
            else:
                subjects = torch.arange(y_pred.shape[0], device=y_pred.device)

            agg_y_pred, agg_groups_pred = agg_retrieval_preds(
                y_pred,
                groups_pred=groups_pred,
                subjects_pred=subjects,
            )
            if retrieval_set_size is not None:
                metric_name += f"_size={retrieval_set_size}"
            else:
                metric_name += f"_size=all"

            metric.reset()
            metric.update(agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true)

            out[metric_name] = metric.compute()

            if "agg" not in metric_name:
                # compute frequency corrected average
                ranks = metric._compute_ranks(
                    agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true
                )
                macro_average = np.mean(
                    list(metric._compute_macro_average(ranks, agg_groups_pred).values())
                )
                out[metric_name + "_macro"] = macro_average
        return out

    #     # log table to wandb if possible
    #     if retrieval_set_size is not None:
    #         metric = Rank()
    #         agg_y_pred, agg_groups_pred = agg_retrieval_preds(
    #             _y_pred,
    #             groups_pred=_groups_pred,
    #             subjects_pred=torch.arange(_y_pred.shape[0], device=_y_pred.device),
    #         )
    #         columns = ["Word", "Rank", "Preds", "Ratios"]
    #         ranks = metric._compute_ranks(
    #             agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true
    #         )
    #         macro_ranks = metric._compute_macro_average(ranks, agg_groups_pred)
    #         pred_labels = metric._get_most_frequent_predictions(
    #             agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true, k=10
    #         )
    #         data = []
    #         for true_label in agg_groups_true:
    #             pred_labels_ = [x[0] for x in pred_labels[true_label]]
    #             counts_ = np.array([x[1] for x in pred_labels[true_label]], dtype=float)
    #             counts_ = [f"{x:.2f}" for x in counts_]
    #             data.append(
    #                 [
    #                     true_label,
    #                     macro_ranks[true_label],
    #                     ",".join(pred_labels_),
    #                     ",".join(counts_),
    #                 ]
    #             )
    #         data = sorted(data, key=lambda x: x[1])  # sort by ranks
    #         if hasattr(pl_module.logger, "log_table"):
    #             pl_module.logger.log_table(
    #                 key=step_name + "_retrieval_rank_" + pl_module.logger.experiment.name,
    #                 columns=columns,
    #                 data=data,
    #             )
