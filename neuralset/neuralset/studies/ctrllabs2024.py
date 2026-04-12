# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty data from CTRL-Labs submitted to the dataset track of NeurIPS 2024.
"""

import typing as tp
from pathlib import Path

import mne
import mne_bids
import pandas as pd

from neuralset.data import BaseData


class Ctrllabs2024(BaseData):
    session: str

    device: tp.ClassVar[str] = "Emg"
    url: tp.ClassVar[str] = "https://www.biorxiv.org/content/10.1101/2024.02.23.581779v2"
    bibtex: tp.ClassVar[str] = ""
    doi: tp.ClassVar[str] = ""
    description: tp.ClassVar[str] = (
        "108 subjects doing surface typing with an EMG wristband on each arm."
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("mne_bids>=0.12",)

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        bpaths = mne_bids.find_matching_paths(root=path, suffixes="eeg")
        for bpath in bpaths:
            if bpath.extension != ".vhdr":
                continue
            yield cls(  # type: ignore
                subject=bpath.subject,
                session=bpath.session,
                path=path,
            )

    def _get_fname(self, suffix: tp.Literal["emg", "events"]) -> Path:
        bids_path = mne_bids.BIDSPath(
            root=self.path,
            subject=self.subject,
            session=self.session,
            task="typing",
            suffix="eeg" if suffix == "emg" else "events",
            datatype="eeg",
            extension=".vhdr" if suffix == "emg" else ".tsv",
        )
        return bids_path.fpath

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        raw = mne.io.read_raw_brainvision(self._get_fname("emg"))
        ch_mapping = {ch_name: "emg" for ch_name in raw.ch_names}
        raw = raw.set_channel_types(ch_mapping)
        return raw

    def _load_events(self) -> pd.DataFrame:
        events = pd.read_csv(self._get_fname("events"), sep="\t")
        events["language"] = "en"
        events["start"] = events.onset

        # Extract sentence events
        sentences = events.trial_type.str.startswith("prompt/")
        events.loc[sentences, "type"] = "Sentence"
        events.loc[sentences, "text"] = events.loc[sentences, "trial_type"].str[
            len("prompt/") : -2
        ]

        # Extract keystroke events
        keystrokes = events.trial_type.str.startswith("key/")
        events.loc[keystrokes, "type"] = "Button"
        events.loc[keystrokes, "text"] = (
            events.loc[keystrokes, "trial_type"]
            .str.replace("key/", "")
            .replace(
                {
                    "Key.backspace": "<backspace>",
                    "Key.enter": "<return>",
                    "Key.space": "<space>",
                }
            )
        )

        uri = f"method:_load_raw?timeline={self.timeline}"
        emg = {"type": "Emg", "filepath": uri, "start": 0}
        events = pd.concat([pd.DataFrame([emg]), events])
        events = events.drop(columns=["trial_type", "onset"])
        events = events.sort_values(by="start")

        return events
