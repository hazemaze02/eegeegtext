# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..data import BaseData


class TestMeg2023(BaseData):
    # study/class level
    device: tp.ClassVar[str] = "Meg"

    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["TestMeg2023"]:
        for i in range(3):
            yield cls(subject=str(i), path=path)

    def _load_events(self) -> pd.DataFrame:
        # FIXME should not be required? use spacy?
        events = pd.DataFrame(
            [
                # use ":" in name to avoid file existence check
                dict(start=1.0, filepath="fake:hello.png", split="train"),
                dict(start=2.0, filepath="fake:world.png", split="train"),
                dict(start=11.0, filepath="fake:good.png", split="test"),
                dict(start=12.0, filepath="fake:bye.png", split="test"),
                dict(
                    start=0,
                    type="Meg",
                    filepath=f"method:_load_raw?timeline={self.timeline}",
                ),
            ]
        )
        events.loc[events.type != "Meg", "type"] = "Image"
        events.loc[events.type == "Image", "duration"] = 0.7
        return events

    def _load_raw(self, timeline: str) -> mne.io.Raw:
        # pylint: disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        fif = Path(self.path) / f"sub-{self.subject}-raw.fif"
        if not fif.exists():
            n_chans = 20
            # lets vary number of channels across subjects
            n_chans += int(self.subject)
            sfreq = 100.0
            n_times = 10_000
            # Use some actual channel names (and some channel names not found in default layout) so
            # we can test out channel position-related features
            layout = mne.channels.read_layout("Vectorview-mag")
            ch_names = layout.names[: n_chans - 1] + ["INVALID_CHANNEL"]
            info = mne.create_info(ch_names, sfreq=sfreq, ch_types="mag")
            np.random.seed(0)
            data = np.ones((n_chans, 1)) * np.arange(n_times)[None, :] / sfreq
            raw = mne.io.RawArray(data, info, verbose=False)
            fif.parent.mkdir(exist_ok=True)
            raw.save(fif, verbose=False)
        return mne.io.Raw(fif, verbose=False)
