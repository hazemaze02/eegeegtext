# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer  # type: ignore
from bids.layout import BIDSFile  # type: ignore

from ..data import BaseData
from ..download import Datalad, Wildcard

logger = logging.getLogger(__name__)

_DEFAULT_BAD_WORDS = frozenset(
    [
        "sentence_start",
        "sentence_end",
        "br",
        "lg",
        "ls",
        "ns",
        "sp",
        "{BR}",
        "{LG}",
        "{LS}",
        "{NS}",
        "{SP}",
    ]
)

_ANAT_TASKS = [
    "AudioMotorLocalizer",
    "AuditoryLocalizer",
    "CategoryLocalizer",
    "MotorLocalizer",
]


def _get_bids_path(
    path: Path | str,
    subject: str,
    session: str,
    task: str,
    layout: tp.Optional[BIDSLayout] = None,
) -> tp.Optional[BIDSFile]:
    if layout is None:
        indexer = BIDSLayoutIndexer(ignore=[".git"])
        layout = BIDSLayout(path, indexer=indexer)

    bids_path = layout.get(
        subject=subject, task=task, session=session, extension="nii.gz", suffix="bold"
    )
    # should only be one possible file path given a subject, session, task
    assert len(bids_path) <= 1
    if len(bids_path) == 0:
        return None
    return bids_path[0]


def _get_audio_file(path: Path | str, task: str) -> Path:
    path = Path(path)
    return path / f"stimuli/{task}.wav"


def _get_audio_text_file(path: Path | str, task: str) -> Path:
    path = Path(path)
    return path / f"derivative/TextGrids/{task}.TextGrid"


def _create_audio_events(path: Path | str, task: str) -> list[dict]:
    # use library provided by deep_fMRI_dataset to process files
    # path of https://github.com/HuthLab/deep-fMRI-dataset/tree/master saved below

    events = []
    dl_path = Path(path) / "download" / "ds003020"
    audio_text_file_name = _get_audio_text_file(dl_path, task)
    audio_wav_file_name = _get_audio_file(dl_path, task)

    split = "train" if task != "wheretheressmoke" else "test"

    events.append(
        dict(
            start=0.0,  # TextGrid file relative to this
            type="Sound",
            language="english",
            modality="Audio",
            filepath=audio_wav_file_name,
            split=split,
        )
    )

    from nltk_contrib.textgrid import TextGrid

    with open(audio_text_file_name, "r", encoding="utf-8") as f:
        data = f.read()
    fid = TextGrid(data)

    for _, tier in enumerate(fid):
        for recording in tier.simple_transcript:
            start, stop, text = recording
            # only add non empty words
            if text != "" and text not in _DEFAULT_BAD_WORDS:
                if tier.nameid == "phone":
                    tier_type = "Phoneme"
                elif tier.nameid == "word":
                    tier_type = "Word"
                else:
                    logger.warning(
                        "Tier must either be phone or word but tier.nameid is %s",
                        tier.nameid,
                    )
                events.append(
                    dict(
                        start=float(start),
                        text=text.lower(),
                        duration=float(stop) - float(start),
                        type=tier_type,
                        language="english",
                        modality="Audio",
                        filepath=audio_wav_file_name,
                        split=split,
                    )
                )

    return events


def _get_preprocessed_responses(path: Path | str, task: str, subject: str) -> np.ndarray:
    # get hf5 data based on https://github.com/HuthLab/deep-fMRI-dataset/blob/eaaa5cd186e0222c374f58adf29ed13ab66cc02a/encoding/encoding_utils.py#L35
    output = _get_response(Path(path) / "download", [task], subject)
    return output


def _get_hf5_path(path: Path | str, subject: str, task: str) -> Path | str | None:
    path = Path(path).resolve()
    hf5_path = path / "derivative" / "preprocessed_data" / f"{subject}" / f"{task}.hf5"
    # should only be one possible file path given a subject, task
    if Path(hf5_path).exists():
        return hf5_path
    return None


def _get_tasks(path: Path):
    path = Path(path).resolve()
    dl_path = path / "stimuli"
    tasks = []
    for fp in Path(dl_path).glob("*.wav"):
        task = str(fp).split("/")
        if len(task) == 0:
            logger.warning(f"Tasks not found for {fp}")
        task_name = task[-1].split(".")[0]
        tasks.append(task_name)
    return tasks


# function from https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/encoding_utils.py
def _get_response(path: Path | str, stories, subject) -> np.ndarray:
    """Get the subject"s fMRI response for stories."""
    import h5py

    path = Path(path).resolve()
    base = path / "ds003020" / "derivative" / "preprocessed_data" / f"{subject}"
    resp = []
    for story in stories:
        resp_path = base / f"{story}.hf5"
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)


class Lebel2023Bold(BaseData):
    session: str
    task: str
    device: tp.ClassVar[str] = "Fmri"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "bids>=0.16.4",
        "nltk>=3.8.1",
        "git+https://github.com/nltk/nltk_contrib.git@683961c53f0c122b90fe2d039fe795e0a2b3e997",  # To read TextGrid files
    )
    TR_FMRI_S: tp.ClassVar[float] = 2.0  # don't rely on nifti header

    @classmethod
    def _download(cls, path: Path) -> None:
        """Download all subjects in the dataset"""
        Datalad(
            study="lebel2023bold",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds003020.git",
            threads=4,
            folders=[
                Wildcard(folder="sub-*"),
                "stimuli",
                Wildcard(folder="derivative/TextGrids/*"),
            ],
        ).download()

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        """
        Iterate over the different recording timelines:
        e.g. subjects x sessions in order with fmri runs
        """
        # dataset details:
        # - 26 stories stimulus + additional story in each session for test dataset
        # - subj1-3 have 55 additional stories + additional test story for total of 82 stories

        path = Path(path)
        dl_dir = path / "download" / "ds003020"
        assert dl_dir.exists(), "run study.download() first"

        indexer = BIDSLayoutIndexer(ignore=[".git"])
        layout = BIDSLayout(dl_dir, indexer=indexer)

        subjects = layout.get_subjects()

        # iterate through all subjects
        for subject in subjects:
            # subj1-3 have maximum 20 sessions, rest of the subjs have 6
            if subject in ["UTS01", "UTS02", "UTS03"]:
                sessions = 20
            else:
                sessions = 6

            # iterate through all sessions for a subject
            for sess in range(1, sessions + 1):
                tasks = sorted(layout.get(return_type="id", target="task", session=sess))

                # iterate through available tasks in a given session
                for task in tasks:
                    if task.startswith(tuple(_ANAT_TASKS)):
                        continue
                    if subject == "UTS01" and sess == 7 and task == "treasureisland":
                        logger.warning(
                            "Skipping subject=UTS01, session=7, task=treasureisland as nii.gz is corrupted."
                        )
                        # See comment on https://openneuro.org/datasets/ds003020/versions/2.2.0
                        continue

                    bids_path = _get_bids_path(
                        path=dl_dir,
                        subject=subject,
                        session=str(sess),
                        task=task,
                        layout=layout,
                    )

                    if bids_path is None or not Path(bids_path.path).exists():
                        logger.warning("Skipping %s", bids_path)
                        continue

                    # check if audio textgrid exists
                    audio_text_file = _get_audio_text_file(path=dl_dir, task=task)
                    assert audio_text_file.exists()
                    # check if stimulus audio exists
                    audio_file = _get_audio_file(path=dl_dir, task=task)
                    assert audio_file.exists()

                    yield cls(path=path, subject=subject, session=str(sess), task=task)  # type: ignore

    def _load_raw(self, timeline: str) -> nibabel.Nifti1Image:
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        """avoid re-reading all the headers"""

        bids_path = _get_bids_path(
            path=Path(self.path) / "download" / "ds003020",
            subject=self.subject,
            session=self.session,
            task=self.task,
        )

        assert bids_path is not None
        assert Path(bids_path.path).exists()

        return bids_path.get_image()

    def _load_events(self) -> pd.DataFrame:
        """Reads the events of a given timeline"""

        # notes:
        # - subj2 data was collected at different location and with a different scan protocol; no localizer data but hand defined ROIs in pycortext-db
        # - subj4 has a missing story scan
        # - subj5 had low visual acuity and was presented auditory cues instead

        nii = self._load_raw(timeline=self.timeline)
        freq = 1.0 / self.TR_FMRI_S
        dur = nii.shape[-1] / freq
        events = _create_audio_events(self.path, self.task)
        uri = f"method:_load_raw?timeline={self.timeline}"
        events.append(
            dict(type="Fmri", start=0.0, filepath=uri, frequency=freq, duration=dur)
        )

        return pd.DataFrame(events)


class LebelProcessed2023Bold(BaseData):
    task: str
    device: tp.ClassVar[str] = "Fmri"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("h5py>=3.10.0",)
    TR_FMRI_S: tp.ClassVar[float] = 2.0  # don't rely on nifti header

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        """
        Iterate over the different recording timelines:
        e.g. subjects x tasks in order with fmri runs
        """
        # dataset details:
        # - each subject has list of story tasks in .hf5

        path = Path(path)
        dl_dir = path / "download" / "ds003020"
        assert dl_dir.exists(), "run study.download() first"

        indexer = BIDSLayoutIndexer(ignore=[".git"])
        layout = BIDSLayout(dl_dir, indexer=indexer)

        subjects = layout.get_subjects()

        # iterate through all subjects
        anat_tasks = tuple(_ANAT_TASKS) + ("auditory_localizer",)
        for subject in subjects:
            tasks = _get_tasks(dl_dir)
            for task in tasks:
                if task.startswith(anat_tasks):
                    continue
                # check if audio textgrid exists
                audio_text_file = _get_audio_text_file(path=dl_dir, task=task)
                if not audio_text_file.exists():
                    msg = f"Missing audio text file for {subject=}, {task=}:\n{audio_text_file}"
                    raise RuntimeError(msg)

                # check if stimulus audio exists
                audio_file = _get_audio_file(path=dl_dir, task=task)
                assert audio_file.exists()
                hf5_path = _get_hf5_path(
                    path=dl_dir,
                    subject=subject,
                    task=task,
                )
                if hf5_path is None or not Path(hf5_path).exists():
                    continue

                yield cls(path=path, subject=subject, task=task)  # type: ignore

    def _load_raw(self, timeline: str) -> nibabel.Nifti2Image:
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        """avoid re-reading all the headers"""

        hf5_path = _get_hf5_path(
            path=Path(self.path) / "download" / "ds003020",
            subject=self.subject,
            task=self.task,
        )

        assert hf5_path is not None
        assert Path(hf5_path).exists()

        hf5_data = _get_preprocessed_responses(
            path=self.path, task=self.task, subject=self.subject
        )

        # flip hf5 data dimensions to be (subject voxels, time steps)
        # since time steps need to be latter dimension
        processed_img = nibabel.Nifti2Image(hf5_data.T, affine=np.eye(4))
        return processed_img

    # get fMRI frequency
    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S

    # get fMRI duration
    def _get_fmri_duration(self) -> float:
        return (
            _get_preprocessed_responses(
                path=self.path, task=self.task, subject=self.subject
            ).shape[0]
            / self._get_fmri_frequency()
        )

    def _load_events(self) -> pd.DataFrame:
        """Reads the events of a given timeline"""

        # notes:
        # - data motion corrected FMRIB Linear Image Registration Tool (FLIRT) from FMRIB Software Library (FSL) 5.0
        # - subj1-3 have 84 story scans (extended dataset due to 10 extra sessions with stories from "The Moth", NYT Modern Love )
        # - subj4 has 26 story scans (missing life.hf5)
        # - subj5-8 have 27 story scans

        # per subj
        # - subj2 data was collected at different location and with a different scan protocol; no localizer data but hand defined ROIs in pycortext-db
        # - subj4 has a missing story scan (missing life.hf5)
        # - subj5 had low visual acuity and was presented auditory cues instead

        events = _create_audio_events(self.path, self.task)
        uri = f"method:_load_raw?timeline={self.timeline}"
        events.append(
            dict(
                type="Fmri",
                frequency=self._get_fmri_frequency(),
                duration=self._get_fmri_duration(),
                start=0.0,
                filepath=uri,
            )
        )

        return pd.DataFrame(events)
