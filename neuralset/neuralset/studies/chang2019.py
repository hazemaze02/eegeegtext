# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
import typing as tp
from pathlib import Path

import nibabel
import pandas as pd

from neuralset import BaseData

# from neuralset.download import Openneuro
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


class Chang2019(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds001499/versions/1.3.1"
    bibtex: tp.ClassVar[
        str
    ] = """
    @dataset{Chang2019,
    author = {Nadine Chang and John A. Pyles and Austin Marcus and Abhinav Gupta and Michael J. Tarr and Elissa M. Aminoff},
    title = {"BOLD5000"},
    year = {2019},
    doi = {10.18112/openneuro.ds001499.v1.3.0},
    publisher = {OpenNeuro}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.18112/openneuro.ds001499.v1.3.0"
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "Preprocessed BOLD data (in MNI152NLin2009aSym) for"
        "4 subjects watching still images in 3T fMRI"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("datalad>=0.19.5",)

    STIMULUS_URL: str = (
        "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
    )

    SESSIONS_PER_SUBJECT: tp.ClassVar[tp.Dict[int, int]] = {
        1: 15,
        2: 15,
        3: 15,
        4: 9,
    }

    # Each session has 9 or 10 runs
    SESSIONS_WITH_10_RUNS: tp.ClassVar[tp.Dict[int, tp.Tuple[int, ...]]] = {
        1: (1, 2, 3, 5, 7, 10, 15),
        2: (1, 3, 4, 5, 11, 12, 14),
        3: (1, 3, 5, 6, 7, 11, 15),
        4: (1, 4, 9),
    }

    BIDS_FOLDER: tp.ClassVar[str] = "download"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives_in_standard_space"

    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"

    TASK: tp.ClassVar[str] = "5000scenes"

    TR_FMRI_S: tp.ClassVar[float] = 2.0

    STIMULI_FOLDER: tp.ClassVar[str] = "BOLD5000_Stimuli/"

    SUBJ_PADDING: tp.ClassVar[str] = "01"
    SUBJ_SUFFIX: tp.ClassVar[str] = "CSI"

    session: int
    run: int

    # FIXME: try using Boto3
    @classmethod
    def _download(cls, path: Path, s3_profile: str = "saml") -> None:
        # Download fMRI data
        subprocess.run(
            (
                f"aws s3 --profile {s3_profile} sync --no-sign-request"
                " s3://openneuro.org/ds001499"
                f" {path}/download"
            ),
            shell=True,
        )

        # Download stimuli data
        stimuli_filename = path / "BOLD5000_Stimuli.zip"
        subprocess.run(["wget", "-O", stimuli_filename, cls.STIMULUS_URL])
        subprocess.run(["unzip", stimuli_filename])

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)

        for subject, session, run in cls._iter_subject_session_run():
            yield cls(subject=subject, session=session, run=run, path=path)

    def _load_events(self) -> pd.DataFrame:
        fmri = {
            "filepath": f"method:_load_raw?timeline={self.timeline}",
            "type": "Fmri",
            "start": 0.0,
            "frequency": self._get_fmri_frequency(),
            "duration": self._get_bold_image().shape[-1] * self.TR_FMRI_S,
        }

        bids_events_df_fp = get_bids_filepath(
            root_path=Path(self.path) / self.BIDS_FOLDER,
            subject=self.subject,
            session=self.session,
            run=self.run,
            task=self.TASK,
            filetype="events",
            data_type="Fmri",
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
        )

        bids_events_df = read_bids_events(bids_events_df_fp)
        repeated_stimuli = self.get_repeated_stimuli()
        image_net_classes = self._get_imagenet_classes()
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df,
            self._get_fmri_frequency(),
            image_net_classes,
            repeated_stimuli,
        )

        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: str) -> nibabel.Nifti1Image:
        return get_masked_bold_image(self._get_bold_image(), self._get_bold_mask())

    @classmethod
    def _iter_subject_session_run(cls):
        for subject in cls.SESSIONS_PER_SUBJECT.keys():
            for session in range(1, cls.SESSIONS_PER_SUBJECT[subject] + 1):
                n_runs = 10 if session in cls.SESSIONS_WITH_10_RUNS[subject] else 9
                for run in range(1, n_runs + 1):
                    yield (subject, session, run)

    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        frequency: float,
        image_net_classes: tp.Dict[str, str],
        repeated_stimuli: tp.List[str],
    ) -> pd.DataFrame:
        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            stimulus_path = self._get_stimulus_path(
                bids_event["stim_file"], bids_event["ImgType"]
            )
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                frequency=frequency,
                filepath=str(
                    Path(self.path)
                    / self.STIMULI_FOLDER
                    / "Scene_Stimuli"
                    / stimulus_path
                ),
                split=(
                    "test" if bids_event["stim_file"] in repeated_stimuli else "train"
                ),
                annotation=self._get_image_annotation(
                    bids_event["stim_file"],
                    bids_event["ImgType"],
                    image_net_classes,
                ),
            )
            ns_events.append(ns_event)

        ns_events_df = pd.DataFrame(ns_events)
        return ns_events_df

    def _get_stimulus_path(self, stim_file, img_type) -> str:
        if img_type.endswith("scenes"):
            return Path("Presented_Stimuli") / "Scene" / stim_file
        elif img_type.endswith("coco"):
            return Path("Presented_Stimuli") / "COCO" / stim_file
        elif img_type.endswith("imagenet"):
            return Path("Presented_Stimuli") / "ImageNet" / stim_file
        else:
            raise ValueError(f"Unknown image type {img_type}")

    def _get_bold_mask(self):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=self.session,
            run=self.run,
            task=self.TASK,
            filetype="bold_mask",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
        )
        return nibabel.load(fp, mmap=True)

    def _get_bold_image(self):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=self.session,
            run=self.run,
            task=self.TASK,
            filetype="bold",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S

    def get_repeated_stimuli(self) -> tp.List[str]:
        return [
            line.strip()
            for line in open(
                Path(self.path)
                / self.STIMULI_FOLDER
                / "Scene_Stimuli"
                / "repeated_stimuli_113_list.txt",
                "r",
            )
        ]

    def _get_imagenet_classes(self) -> tp.Dict[str, str]:
        image_to_annotations = {}
        with open(
            Path(self.path)
            / self.STIMULI_FOLDER
            / "Image_Labels"
            / "imagenet_final_labels.txt",
            "r",
        ) as file:
            for line in file:
                identifier, string_list = line.split(" ", 1)
                image_to_annotations[identifier] = " ".join(
                    [s.strip() for s in string_list.split(",")]
                )
        return image_to_annotations

    def _get_image_annotation(
        self, stim_file: str, img_type: str, image_net_classes: tp.Dict[str, str]
    ) -> str:
        if img_type in ["scenes", "rep_scenes"]:
            match = re.match(r"([a-zA-Z_]+)(\d|[.])", stim_file)
            if match is not None:
                return match.group(1)
            else:
                raise ValueError(f"'{stim_file}' has an unexpected pattern")
        elif img_type in ["imagenet", "rep_imagenet"]:
            return image_net_classes[stim_file.split("_")[0]]
        elif img_type in ["coco", "rep_coco"]:
            return "none"
        else:
            raise ValueError(
                f"{img_type} should be one of 'scenes', 'imagenet', or 'coco'"
            )
