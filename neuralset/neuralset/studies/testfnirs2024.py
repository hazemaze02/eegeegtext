# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..data import BaseData


class TestFnirs2024(BaseData):
    """Clone of TestMeg2023 but adapted for testing fNIRS features."""

    # study/class level
    device: tp.ClassVar[str] = "Fnirs"

    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["TestFnirs2024"]:
        for i in range(3):
            yield cls(subject=str(i), path=path)

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # pylint: disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        fname = Path(self.path) / f"sub-{self.subject}-raw.fif"

        if not fname.exists():
            # E.g. Artinis Octamon has 8 sources and 2 detectors
            n_sources, n_detectors = 8, 2
            wavelengths = [760, 850]
            ch_names = [
                f"S{source + 1}_D{detector + 1} {wl}"
                for source, detector, wl in product(
                    range(n_sources), range(n_detectors), wavelengths
                )
            ]
            ch_types = ["fnirs_cw_amplitude"] * len(ch_names)
            sfreq = 10.0
            n_times = int(sfreq * 5 * 60)
            info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
            np.random.seed(0)
            data = np.ones((len(ch_names), 1)) * np.arange(n_times)[None, :] / sfreq
            raw = mne.io.RawArray(data, info, verbose=False)

            # TODO: Enable the following so we can test distance-based processing
            # montage = mne.channels.make_standard_montage('artinis-octamon')
            # raw.set_montage(montage)

            fname.parent.mkdir(exist_ok=True)
            raw.save(fname, verbose=False)

        return mne.io.read_raw_fif(fname)

    def _load_events(self) -> pd.DataFrame:
        sentences = "hello world. the quick brown fox. they quit. good bye."

        events = []
        start = 1.0
        splits = ["train", "test", "val"]
        for sid, sentence in enumerate(sentences.split(".")):
            if not sentence:
                continue
            sentence += "."
            for word in sentence.split():
                events.append(
                    dict(
                        start=start,
                        text=word,
                        duration=len(word) / 30,
                        type="Word",
                        language="english",
                        modality="audio",
                        split=splits[sid % 3],
                    )
                )
                start += 0.5
            start += 2.0
        uri = f"method:_load_raw?timeline={self.timeline}"
        events.append({"type": "Fnirs", "filepath": uri, "start": 0})
        return pd.DataFrame(events)
