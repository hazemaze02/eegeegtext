# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Task-based functional MRI where participants view images of objects depicted as photographs,
line drawings, or sketchs. A subset of participants  completed an adapted version of the task
with MEG. Data was not available for participant 3 (no data) and 23 (no anatomical scans).

Image events were left out because we couldn't identify the correct the image-event mapping.
These should be added if/when this information is known.
... Enumerating based on alphabetical order produced at-chance results.
"""

import typing as tp
import warnings
from pathlib import Path

import nibabel
import pandas as pd

from neuralset import BaseData
from neuralset.download import Datalad, Osf, Wildcard
from neuralset.utils import get_bids_filepath


class Singer2023Bold(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004331/versions/1.0.4"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Singer2023Bold,
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
        "BOLD data for 30 subjects watching still images (photos, drawings, and sketches) in 3T fMRI"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
    )

    SUBJECTS: tp.ClassVar[tp.Tuple[int, ...]] = tuple(range(1, 32))

    N_RUNS: tp.ClassVar[int] = 12

    BIDS_FOLDER: tp.ClassVar[str] = "download/ds004331"
    STIMULUS_FOLDER: tp.ClassVar[str] = "download/stimuli"

    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives_output"

    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym_res-1"

    TASK: tp.ClassVar[str] = "main"

    TR_FMRI_S: tp.ClassVar[float] = 1.5

    run: int

    @classmethod
    def _download(cls, path: Path) -> None:
        # Download neuroimaging data from Datalad
        Datalad(
            study="singer2023bold",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds004331.git",
            threads=4,
            folders=[
                Wildcard(folder="sub-*"),
            ],
        ).download()

        # Download stimuli from Osf
        Osf(study="cebhv", dset_dir=path, folder="download/stimuli").download()  # type: ignore

    @classmethod
    def _iter_subject_run(cls):
        for subject in cls.SUBJECTS:
            # Subject 3 missing all data, Subject 23 missing anat (T1)
            if subject in [3, 23]:
                continue
            for run in range(1, cls.N_RUNS + 1):
                yield (subject, run)

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)

        for subject, run in cls._iter_subject_run():
            yield cls(subject=subject, run=run, path=path)

    def _load_events(self) -> pd.DataFrame:
        fmri = {
            "filepath": f"method:_load_raw?timeline={self.timeline}",
            "type": "Fmri",
            "start": 0.0,
            "frequency": self._get_fmri_frequency(),
            "duration": self._get_bold_image().shape[-1] * self.TR_FMRI_S,
        }

        warnings.warn("Image stimuli NOT loaded in events. File mapping unknown.")

        return pd.DataFrame([fmri])

    # TODO: Find correct image event mapping.
    # Enumerating images assuming alphabetical order produced at-chance results.
    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        event_fn_map: dict[str, str],
        frequency: float,
    ) -> pd.DataFrame:
        # Leave out 'catch' trials (used for making sure subject is focused)
        bids_events_df = bids_events_df[bids_events_df.trial_type != "Catch"]
        ns_events_df = bids_events_df[["onset", "duration"]].copy()
        ns_events_df.insert(0, "type", "Image")
        ns_events_df.insert(3, "frequency", frequency)
        ns_events_df["filepath"] = bids_events_df["trial_type"].replace(event_fn_map)
        ns_events_df.columns = pd.Index(
            ["type", "start", "duration", "frequency", "filepath"]
        )
        return ns_events_df

    def _load_raw(self, timeline: str):
        return self._get_bold_image()

    def _get_raw_bold_image(self):
        fp = get_bids_filepath(
            root_path=Path(self.path) / self.BIDS_FOLDER,
            subject=self.subject,
            session=None,
            run=self.run,
            task=self.TASK,
            filetype="bold_raw",
            data_type="Fmri",
            run_padding="",
            run_suffix="00",
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S

    def _get_bold_mask(self):
        fp = get_bids_filepath(
            root_path=Path(self.path) / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=None,
            run=self.run,
            task=self.TASK,
            filetype="bold_mask",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            run_padding="",
            run_suffix="00",
        )
        return nibabel.load(fp, mmap=True)

    def _get_bold_image(self):
        fp = get_bids_filepath(
            root_path=Path(self.path) / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=None,
            run=self.run,
            task=self.TASK,
            filetype="bold",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            run_padding="",
            run_suffix="00",
        )
        return nibabel.load(fp, mmap=True)
