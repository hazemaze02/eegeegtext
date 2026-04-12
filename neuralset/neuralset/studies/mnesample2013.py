# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import mne
import pandas as pd

from ..data import BaseData


class MneSample2013(BaseData):
    subject: str = "sample"
    version: tp.ClassVar[str] = "v3"

    # Study level
    device: tp.ClassVar[str] = "Meg"
    url: tp.ClassVar[str] = "https://mne.tools/stable/overview/datasets_index.html#sample"
    description: tp.ClassVar[str] = """mne sample MEG dataset"""

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(cls, path: Path | str) -> tp.Iterator["MneSample2013"]:
        yield cls(path=path)

    def _load_events(self) -> pd.DataFrame:
        # read raw
        path = Path(self.path).absolute()
        data_path = mne.datasets.sample.data_path(path) / "MEG" / "sample"
        raw_fname = data_path / "sample_audvis_filt-0-40_raw.fif"
        raw = mne.io.read_raw_fif(raw_fname, verbose=False)
        freq = raw.info["sfreq"]
        start = raw.first_samp / freq  # account for first_samp

        # find stimulus events
        events = mne.find_events(raw)
        df = pd.DataFrame(events, columns=["start_idx", "duration_idx", "trigger"])
        df["start"] = df.start_idx / raw.info["sfreq"]
        df["duration"] = 1.0 / raw.info["sfreq"]
        df["type"] = "Stimulus"
        df.modality = None
        df.modality = None
        df = df.loc[df.trigger <= 4].reset_index()
        df.loc[df.trigger <= 2, "modality"] = "audio"
        df.loc[(df.trigger == 3) | (df.trigger == 4), "modality"] = "visual"
        df.loc[(df.trigger == 1) | (df.trigger == 3), "side"] = "left"
        df.loc[(df.trigger == 2) | (df.trigger == 4), "side"] = "right"
        df["code"] = df.trigger.map(lambda x: x - 1)
        df["description"] = df.side + "_" + df.modality

        # meg event
        meg = {"type": "Meg", "filepath": raw_fname, "start": start, "frequency": freq}
        df = pd.concat([pd.DataFrame([meg]), df])
        return df
