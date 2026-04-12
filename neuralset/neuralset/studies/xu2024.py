# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NOTE: The data is currently available as `.fif` files, however the GitHub documentation suggests
there exists a BIDS version as well. If this becomes available, it might make sense to rewrite the
implementation to use the BIDS version instead.
"""

import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralset.data import BaseData

from ..download import Osf


class Xu2024(BaseData):
    subject: str
    session: int

    # study/class level
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://osf.io/kqgs8"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{xu2024alljoined,
        title={Alljoined--A dataset for EEG-to-Image decoding},
        author={Xu, Jonathan and Aristimunha, Bruno and Feucht, Max Emanuel and Qian, Emma and Liu,
        Charles and Shahjahan, Tazik and Spyra, Martyna and Zhang, Steven Zifan and Short, Nicholas
        and Kim, Jioh and others},
        journal={arXiv preprint arXiv:2404.05553},
        year={2024}
    }
    """
    doi: tp.ClassVar[str] = "UNKNOWN"
    licence: tp.ClassVar[str] = "UNKNOWN"
    description: tp.ClassVar[str] = "8 subjects watching static images in EEG."
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("h5py",)

    _nsd_stimuli_path: Path | None = None

    @classmethod
    def _download(cls, path: Path) -> None:
        Osf(study="kqgs8", dset_dir=path, folder="xu2024").download()

    @staticmethod
    def _get_fname(
        path: str | Path,
        subject: str,
        session: int,
        kind: tp.Literal["raw", "epochs", "h5"] = "raw",
    ):
        if kind == "raw":
            folder, suffix = "raw", "_eeg.fif"
        elif kind == "epochs":
            folder, suffix = "raw", "_epo.fif"
        elif kind == "h5":
            folder, suffix = "05_125", ".h5"
        return Path(path) / folder / f"subj{int(subject):02}_session{session}{suffix}"

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        """Returns a generator of all recordings"""
        for subject, session in product(range(1, 9), range(1, 3)):
            fname = cls._get_fname(path, str(subject), session)  # type: ignore[arg-type]
            if fname.exists():
                yield cls(subject=str(subject), session=session, path=path)  # type: ignore[arg-type]

    def _get_nsd_stimuli_path(self) -> Path:
        return (Path(self.path) / ".." / "allen2022bold" / "nsd_stimuli").resolve()

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # Necessary to ensure montage information is available in the Raw object
        filepath = str(self._get_fname(self.path, self.subject, self.session, kind="raw"))
        raw = mne.io.read_raw(filepath)
        raw.set_montage("standard_1020")
        return raw

    def _load_events(self) -> pd.DataFrame:
        """
        Broken/missing files:
        - subj02, session 2
        - subj03, session 1
        - subj07, session 2
        - subj08, session 2
        """
        if self.subject == "3" and self.session == 1:  # Missing (22/05/24)
            return pd.DataFrame()

        # Load image event information
        h5_fname = self._get_fname(self.path, self.subject, self.session, kind="h5")
        events = pd.read_hdf(h5_fname).drop("eeg", axis=1)
        events["filepath"] = events["73k_id"].apply(
            lambda x: str(self._get_nsd_stimuli_path() / f"{x}.png")
        )
        events["start"] = events.curr_time
        events["duration"] = 0.3
        events["type"] = "Image"

        events = events.drop(columns=["subject_id", "session", "curr_time"])

        eeg = {
            "type": "Eeg",
            "start": 0.0,
            "filepath": f"method:_load_raw?timeline={self.timeline}",
        }
        events = pd.concat([pd.DataFrame([eeg]), events])  # type: ignore

        return events
