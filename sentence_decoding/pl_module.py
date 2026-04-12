# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
import warnings

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn, optim

from neuralset.dataloader import SegmentData

logger = logging.getLogger(__name__)


class BrainModule(pl.LightningModule):
    """
    Torch-lightning module for M/EEG model training.

    N400 Integration (Task 2)
    -------------------------
    When ``use_n400=True``, the module extracts two cognitive features from
    the raw EEG signal on every forward pass:

      • N400 amplitude  — mean voltage at centro-parietal channels in the
                          300–500 ms window after word onset.  Larger (more
                          positive after z-scoring) = semantically harder word.
      • Theta power     — 4–8 Hz band power, linked to working memory load.

    These two scalars are concatenated to the brain embedding vector produced
    by the EEG encoder, and then projected back to the original embedding
    dimension via a small ``nn.Linear(brain_dim + 2, brain_dim)`` layer.
    This keeps the rest of the architecture (transformer, loss, metrics)
    completely unchanged.

    Architecture (with N400):
        EEG (B, C, T)
            │
            ▼  EEG encoder (e.g. SimpleConvTimeAgg)
        brain_embed  (B, brain_dim)
            │
            ├── N400 extraction from raw EEG → (B, 2)
            │
            ▼  concat → (B, brain_dim + 2)
            │
            ▼  n400_projection  Linear(brain_dim + 2, brain_dim)
            │
        brain_embed  (B, brain_dim)   ← same shape as before
            │
            ▼  (transformer + loss unchanged)

    Set ``use_n400=False`` to disable and ablate this addition.
    """

    def __init__(
        self,
        model,
        transformer,
        loss,
        metrics,
        retrieval_metrics,
        trainer_config,
        target_scaler=None,
        checkpoint_path=None,
        # ----------------------------------------------------------------
        # N400 integration parameters (Task 2)
        # ----------------------------------------------------------------
        use_n400: bool = False,
        channel_names: tp.Optional[tp.List[str]] = None,
        eeg_sfreq: float = 50.0,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            The EEG encoder (e.g. SimpleConvTimeAgg).
        transformer : nn.Module or None
            Optional sentence-level transformer.
        loss, metrics, retrieval_metrics, trainer_config, target_scaler,
        checkpoint_path : (unchanged from original)

        use_n400 : bool
            If True, extract N400 + theta features from the raw EEG signal
            and inject them into the brain embedding.  Set to False to run
            the model in its original form (useful for ablation studies).

        channel_names : list of str or None
            EEG channel names in the same order as the channel dimension of
            the input tensor.  Used to identify centro-parietal channels for
            N400 extraction.  If None, all channels are averaged (fallback).
            For TestEeg2024 this would be e.g. ['Fp1', 'Fp2', ..., 'Cz', 'Pz', ...]

        eeg_sfreq : float
            Sampling frequency of the EEG tensor arriving in the batch
            (after the pipeline's resampling step).  Default 50 Hz matches
            the ``"frequency": 50.0`` setting in grids/defaults.py.
        """
        super().__init__()
        self.model = model
        self.transformer = transformer
        self.trainer_config = trainer_config

        self.target_scaler = target_scaler
        self.checkpoint_path = checkpoint_path

        self.loss = loss
        self.metrics = nn.ModuleDict(
            {split + "_" + k: v for k, v in metrics.items() for split in ["val", "test"]}
        )
        self.retrieval_metrics = nn.ModuleDict(
            {
                split + "_" + k: v
                for k, v in retrieval_metrics.items()
                for split in ["val", "test"]
            }
        )

        # ------------------------------------------------------------------
        # N400 integration — projection layer (Task 2)
        # ------------------------------------------------------------------
        self.use_n400 = use_n400
        self.channel_names = channel_names      # stored for N400 channel lookup
        self.eeg_sfreq = eeg_sfreq              # Hz — needed to convert ms → samples

        # The projection layer maps (brain_dim + 2) → brain_dim so that the
        # rest of the architecture is completely unaffected.
        # We create it only when use_n400=True to avoid any overhead otherwise.
        self.n400_projection: tp.Optional[nn.Linear] = None
        if use_n400:
            brain_dim = self._get_model_output_dim(model)
            if brain_dim is not None:
                # Linear(brain_dim + 2, brain_dim) with bias
                # The +2 accounts for the N400 scalar and the theta scalar.
                self.n400_projection = nn.Linear(brain_dim + 2, brain_dim)
                logger.info(
                    "[N400] Created N400 projection layer: "
                    f"Linear({brain_dim} + 2, {brain_dim})"
                )
                print(
                    f"[N400] N400 integration enabled.  "
                    f"Projection: Linear({brain_dim} + 2 → {brain_dim})"
                )
            else:
                warnings.warn(
                    "[N400] Could not determine brain embedding dimension from model.  "
                    "N400 injection is DISABLED.  Supported model types: SimpleConv, "
                    "SimpleConvTimeAgg (have .out_channels), EEGNet (has .n_outputs)."
                )
                self.use_n400 = False

    @staticmethod
    def _get_model_output_dim(model: nn.Module) -> tp.Optional[int]:
        """
        Return the output embedding dimension of the EEG encoder.

        We look for the convention used by each model class in neuraltrain:
          • SimpleConv / SimpleConvTimeAgg → model.out_channels
          • EEGNet                         → model.n_outputs

        Returns None if the dimension cannot be determined automatically.
        """
        if hasattr(model, "out_channels"):
            return model.out_channels
        if hasattr(model, "n_outputs"):
            return model.n_outputs
        return None

    def cnn_forward(self, batch):
        """
        Run the EEG encoder and (optionally) inject N400 features.

        Flow:
          1. Run the raw EEG through the CNN encoder (SimpleConv / EEGNet / etc.)
          2. If use_n400=True and the output is 2-D (time-aggregated):
               a. Extract N400 amplitude and theta power from the raw EEG signal.
               b. Concatenate the two scalars to the brain embedding.
               c. Project back to the original brain_dim via n400_projection.

        The N400 injection only applies when the encoder already performs
        temporal aggregation (output is 2-D: batch × brain_dim).  For 3-D
        outputs (batch × brain_dim × time) — which occur with plain SimpleConv
        without time aggregation — the injection is skipped and a warning is
        printed.  In the standard test config, SimpleConvTimeAgg is used, which
        always outputs 2-D after its attention pooling step.
        """
        x = batch.data["neuro"]   # shape: (batch_size, n_channels, n_timepoints)

        subject_ids = batch.data["subject_id"] if "subject_id" in batch.data else None
        channel_positions = (
            batch.data["channel_positions"] if "channel_positions" in batch.data else None
        )

        # ------------------------------------------------------------------
        # Standard CNN forward (unchanged)
        # ------------------------------------------------------------------
        model_name = self.model.__class__.__name__
        if "SimpleConv" in model_name:
            y_pred = self.model(x, subject_ids, channel_positions)
        elif model_name == "EEGNet":
            y_pred = self.model(x)
        elif model_name in ["LinearModel"]:
            y_pred = self.model(x, subject_ids)
        else:
            raise ValueError(f"Unknown model {model_name}")

        # ------------------------------------------------------------------
        # N400 feature injection (Task 2) — only when use_n400=True
        # ------------------------------------------------------------------
        if self.use_n400 and self.n400_projection is not None:

            if y_pred.ndim != 2:
                # N400 injection requires a 2-D brain embedding (i.e. a model
                # that already aggregates across time, like SimpleConvTimeAgg).
                # For 3-D outputs, skip silently — they will be reshaped later
                # in _run_step, but the semantics of injecting a per-epoch scalar
                # into a time-resolved representation are ill-defined.
                if not getattr(self, "_n400_warned_3d", False):
                    warnings.warn(
                        "[N400] Model output is 3-D (batch, dim, time).  "
                        "N400 injection is only supported for time-aggregated "
                        "(2-D) outputs.  Use SimpleConvTimeAgg with "
                        "time_agg_out='gap' or 'att'.  Skipping N400 for now."
                    )
                    self._n400_warned_3d = True   # warn only once per run
            else:
                # Extract N400 features from the raw EEG (CPU numpy array)
                # Shape of x: (batch_size, n_channels, n_timepoints)
                #
                # NOTE: We call .detach() so that this numpy conversion does
                # not interfere with gradient computation.  The N400 features
                # are treated as fixed inputs (not learned from), similarly to
                # how positional encodings are fixed in a transformer.
                x_np = x.detach().cpu().numpy()

                from sentence_decoding.n400_pipeline import compute_n400_from_batch
                n400_np = compute_n400_from_batch(
                    x_np,
                    channel_names=self.channel_names,
                    sfreq=self.eeg_sfreq,
                )
                # n400_np shape: (batch_size, 2)
                # Column 0: z-scored N400 amplitude
                # Column 1: z-scored theta power

                # Move to the same device as y_pred (GPU/CPU)
                n400_tensor = torch.from_numpy(n400_np).to(y_pred.device)

                # Concatenate: (batch_size, brain_dim) + (batch_size, 2)
                #            → (batch_size, brain_dim + 2)
                y_pred_aug = torch.cat([y_pred, n400_tensor], dim=1)

                # Project back to brain_dim
                # Linear(brain_dim + 2, brain_dim) — learned during training.
                # This preserves the output shape so the transformer and loss
                # do not need any modification.
                y_pred = self.n400_projection(y_pred_aug)

        return y_pred

    def transformer_forward(self, batch, y_pred):
        sentence_uids = np.array(
            [
                f"{segment._trigger['sequence_id']}_{segment._trigger['timeline']}"
                for segment in batch.segments
            ]
        )

        # pad and group according to sentences
        unique_uids, sentence_idx = np.unique(sentence_uids, return_index=True)
        unique_uids = unique_uids[
            np.argsort(sentence_idx)
        ]  # beware of order in np.unique!!!
        grouped_y_pred = []
        for uid in unique_uids:
            indices = [i for i, s in enumerate(sentence_uids) if s == uid]
            grouped_y_pred.append(torch.stack([y_pred[i] for i in indices]))
        max_len = max([len(y) for y in grouped_y_pred])

        # pad for transformer
        transformer_input = torch.zeros(len(grouped_y_pred), max_len, y_pred.shape[1]).to(
            y_pred.device
        )
        mask = torch.zeros(len(grouped_y_pred), max_len).to(y_pred.device)
        for i, y in enumerate(grouped_y_pred):
            transformer_input[i, : len(y)] = y
            mask[i, : len(y)] = 1

        # feed to transformer
        transformer_output = self.transformer(transformer_input, mask=mask.bool())

        # unpad and ungroup
        out = []
        for i, y in enumerate(grouped_y_pred):
            out.extend(transformer_output[i][: len(y)])
        out = torch.stack(out)

        out = out / out.norm(dim=1, keepdim=True)

        return out

    def _run_step(self, batch, step_name):

        y_true = batch.data["feature"]
        if self.target_scaler is not None:
            y_true = self.target_scaler.transform(y_true)

        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_true.shape[0],
        }

        # SimpleConv processing
        y_pred = self.cnn_forward(batch)
        if len(y_pred.shape) == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_true = y_true.reshape(y_true.shape[0], -1)

        y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
        loss = self.loss(y_pred, y_true)
        self.log(f"{step_name}_cnn_loss", loss, **log_kwargs)

        # Transformer processing
        if (
            self.transformer is not None
            and self.current_epoch >= self.trainer_config.transformer_start_epoch
        ):
            y_transformer = self.transformer_forward(batch, y_pred)
            transformer_loss = self.loss(y_transformer, y_true)

            self.log(f"{step_name}_transformer_loss", transformer_loss, **log_kwargs)
            loss = transformer_loss
            y_pred = y_transformer

        # Compute metrics
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(metric_name, metric, **log_kwargs)

        return loss, y_pred, y_true

    def training_step(self, batch: SegmentData, batch_idx: int, dataloader_idx: int = 0):
        loss, _, _ = self._run_step(batch, step_name="train")
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, step_name="val")
        return y_pred, y_true

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, step_name="test")
        return y_pred, y_true

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.trainer_config.lr,
            weight_decay=self.trainer_config.weight_decay,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer_config.n_epochs
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
