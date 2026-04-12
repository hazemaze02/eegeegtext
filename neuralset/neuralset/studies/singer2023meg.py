# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Task-based functional MRI where participants view images of objects depicted as photographs,
line drawings, or sketchs. A subset of participants  completed an adapted version of the task
with fMRI.

Data not avaiable for participant 3.

Image events were left out because we couldn't identify the correct the image-event mapping.
These should be added if/when this information is known.
... Enumerating based on alphabetical order produced at-chance results.
"""

import typing as tp
import warnings
from pathlib import Path

import mne
import mne_bids
import pandas as pd

from neuralset import BaseData
from neuralset.download import Datalad, Osf, Wildcard


class Singer2023Meg(BaseData):
    device: tp.ClassVar[str] = "Meg"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004330/versions/1.0.0"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Singer2023Meg,
        title={The Spatiotemporal Neural Dynamics of Object Recognition for Natural Images and Line Drawings},
        author={Singer, Johannes JD and Cichy, Radoslaw M and Hebart, Martin N},
        journal={The Journal of Neuroscience},
        volume={43},
        number={3},
        pages={484–-500},
        year={2023},
        publisher={Society for Neuroscience},
    }
    """
    doi: tp.ClassVar[str] = "doi:10.1523/JNEUROSCI.1546-22.2022"

    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "MEG data for 30 subjects watching still images (photos, drawings, and sketches)"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
    )

    SUBJECTS: tp.ClassVar[tp.Tuple[str, ...]] = tuple(
        [f"{int(i):{'02'}}" for i in range(1, 32)]
    )
    RUNS: tp.ClassVar[tp.Tuple[str, ...]] = tuple(
        [f"{int(i):{'02'}}" for i in range(1, 10)]
    )
    SESSION: tp.ClassVar[str] = "01"
    TASK: tp.ClassVar[str] = "main"
    BIDS_FOLDER: tp.ClassVar[str] = "download/ds004330"
    STIMULUS_FOLDER: tp.ClassVar[str] = "download/stimuli"

    run: str
    session: str

    @classmethod
    def _download(cls, path: Path) -> None:
        # Download neuroimaging data from Datalad
        Datalad(
            study="singer2023meg",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds004330.git",
            threads=4,
            folders=[
                Wildcard(folder="sub-*"),
            ],
        ).download()
        # Download stimuli from Osf
        Osf(study="cebhv", dset_dir=path, folder="download/stimuli").download()  # type: ignore

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)
        for subject in cls.SUBJECTS:
            if subject == "03":
                continue
            for run in cls.RUNS:
                yield cls(subject=subject, session=cls.SESSION, run=run, path=path)

    def _load_events(self) -> pd.DataFrame:
        meg = {
            "filepath": f"method:_load_raw?timeline={self.timeline}",
            "type": "Meg",
        }
        meg_info = self._get_meg_info()

        warnings.warn("Image stimuli NOT loaded in events. File mapping unknown.")

        return pd.DataFrame([{**meg, **meg_info}])

    # TODO: Find correct image event mapping.
    # Enumerating images assuming alphabetical order produced at-chance results.
    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        event_fn_map: dict,
    ) -> pd.DataFrame:
        ns_events_df = bids_events_df[["onset", "duration"]].copy()
        ns_events_df.insert(0, "type", "Image")
        ns_events_df["filepath"] = bids_events_df["trial_type"].replace(event_fn_map)
        ns_events_df.columns = pd.Index(["type", "start", "duration", "filepath"])
        return ns_events_df

    def _load_raw(self, timeline: str) -> mne.io.Raw:
        fname = self._get_filename(suffix="meg")
        return mne.io.read_raw_fif(fname)

    def _get_filename(self, suffix: tp.Literal["meg", "events"]) -> Path:
        data_path = Path(self.path) / "download" / "ds004330"
        return mne_bids.BIDSPath(
            root=data_path,
            subject=self.subject,
            session=self.SESSION,
            run=self.run.zfill(2),
            task=self.TASK,
            suffix="meg" if suffix == "meg" else "events",
            datatype="meg",
            extension=".fif" if suffix == "meg" else ".tsv",
        )

    def _get_meg_info(self) -> dict:
        fname = self._get_filename(suffix="meg")
        meg_data = mne.io.read_raw_fif(fname)
        duration = meg_data.n_times / 1000.0
        start = meg_data.first_samp / 1000.0
        frequency = meg_data.info["sfreq"]
        return {"start": start, "duration": duration, "frequency": frequency}
