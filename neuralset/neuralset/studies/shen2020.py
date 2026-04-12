# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import subprocess
import typing as tp
from pathlib import Path

import nibabel
import pandas as pd
import requests
from tqdm import tqdm

from neuralset import BaseData
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


def _read_tsv_to_dict(filename: str | Path) -> tp.Dict[str, str]:
    result_dict = {}
    with open(filename, "r") as file:
        for line in file:
            name, item_id = line.strip().split("\t")[:2]
            result_dict[item_id] = name
    return result_dict


def _copy_imagenet_images(
    tsv_path: str | Path, destination_folder: str | Path, imagenet_22k_folder: str | Path
) -> None:
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    with open(tsv_path, "r") as file:
        lines = file.readlines()

        for line in tqdm(lines, desc="Copying ImageNet images"):
            identifier = line.strip().split("\t")[0]

            prefix = identifier.split("_")[0]
            source_folder = Path(imagenet_22k_folder) / prefix
            source_file_path = source_folder / (identifier + ".JPEG")

            destination_file_path = destination_folder / (identifier + ".JPEG")

            if not source_folder.exists():
                raise FileNotFoundError(f"No such folder: {source_folder}")

            if not source_file_path.is_file():
                raise FileNotFoundError(f"No such file: {source_file_path}")

            shutil.copy2(source_file_path, destination_file_path)


class Shen2020(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds001506/versions/1.3.1"
    bibtex: tp.ClassVar[
        str
    ] = """
        @dataset{Shen2020,
        author = {Guohua Shen and Tomoyasu Horikawa and Kei Majima and Yukiyasu Kamitani},
        title = {"Deep Image Reconstruction"},
        year = {2020},
        doi = {10.18112/openneuro.ds001506.v1.3.1},
        publisher = {OpenNeuro}
        }
    """
    doi: tp.ClassVar[str] = "doi:10.18112/openneuro.ds001506.v1.3.1"
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "Preprocessed BOLD data (in MNI152NLin2009aSym) for"
        "3 subjects watching still images (natural, artificial, and letters) in 3T fMRI"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()

    BIDS_FOLDER: tp.ClassVar[str] = "ds001506-download"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives"

    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"

    TR_FMRI_S: tp.ClassVar[float] = 2.0

    STIMULI_FOLDER: tp.ClassVar[str] = "stimuli"

    SUBJ_PADDING: tp.ClassVar[str] = "02"

    SUBJECTS: tp.ClassVar[tp.Tuple[int, ...]] = (1, 2, 3)

    TASK: tp.ClassVar[str] = "perception"

    RUN_DESIGN: tp.ClassVar[tp.Dict] = {
        "perceptionNaturalImageTraining": {
            "n_sessions": {1: 15, 2: 15, 3: 15},
            "n_runs": {
                1: {7: 4, 8: 10, 9: 10, "default": 8},
                2: {"default": 8},
                3: {"default": 8},
            },
        },
        "perceptionNaturalImageTest": {
            "n_sessions": {1: 3, 2: 3, 3: 3},
            "n_runs": {
                1: {"default": 8},
                2: {1: 10, 2: 6, "default": 8},
                3: {"default": 8},
            },
        },
        "perceptionArtificialImage": {
            "n_sessions": {1: 2, 2: 3, 3: 2},
            "n_runs": {
                1: {"default": 10},
                2: {1: 10, 2: 8, 3: 2},
                3: {"default": 10},
            },
        },
    }

    session_type: str
    session: int
    run: int

    @classmethod
    def _download(
        cls,
        path: Path,
        path_to_imagenet_22k: str = "/datasets01/imagenet-22k/062717",
        s3_profile: str = "saml",
    ) -> None:
        """
        path_to_imagenet_22k: path to a folder containing Imagenet22k dataset, that is:
        for each synset 'nX', it has a subfolder 'nX' containing all ImageNet images
        for that synset, and the name of each image in this folder
        is of the form 'nX_{img_id}.JPEG'
        """
        # Download fMRI data
        subprocess.run(
            [
                "aws",
                "s3",
                "--profile",
                f"{s3_profile}",
                "sync",
                "--no-sign-request",
                "s3://openneuro.org/ds001506",
                str(path / "ds001506-download/"),
            ],
            check=True,
        )

        # Create stimuli folder
        path_to_stimuli = path / cls.STIMULI_FOLDER
        path_to_stimuli.mkdir(parents=True, exist_ok=True)

        # Download stimuli id / name CSVs
        data = {
            "perceptionNaturalImageTraining.tsv": "https://ndownloader.figshare.com/files/14876741",
            "perceptionNaturalImageTest.tsv": "https://ndownloader.figshare.com/files/14876738",
            "perceptionLetterImage.tsv": "https://ndownloader.figshare.com/files/14876732",
            "perceptionArtificialImage.tsv": "https://ndownloader.figshare.com/files/14876798",
        }

        for name, url in data.items():
            with open(path_to_stimuli / name, "wb") as f:
                f.write(requests.get(url).content)

        # Use ImageNet-22k path to retrieve natural stimuli in correct folder
        _copy_imagenet_images(
            path_to_stimuli / "perceptionNaturalImageTraining.tsv",
            path_to_stimuli / "perceptionNaturalImageTraining",
            path_to_imagenet_22k,
        )
        _copy_imagenet_images(
            path_to_stimuli / "perceptionNaturalImageTest.tsv",
            path_to_stimuli / "perceptionNaturalImageTest",
            path_to_imagenet_22k,
        )

        # Download artificial and letter stimuli and unzip them
        custom_stimulus_data = {
            "perceptionArtificialImage.zip": "https://ndownloader.figshare.com/files/14876801",
            "perceptionLetterImage.zip": "https://ndownloader.figshare.com/files/14876735",
        }
        for name, url in custom_stimulus_data.items():
            with open(path_to_stimuli / name, "wb") as f:
                f.write(requests.get(url).content)

        subprocess.run(
            [
                "unzip",
                str(path_to_stimuli / "perceptionArtificialImage.zip"),
                "-d",
                str(path_to_stimuli / "perceptionArtificialImage"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "unzip",
                str(path_to_stimuli / "perceptionLetterImage.zip"),
                "-d",
                str(path_to_stimuli / "perceptionLetterImage"),
            ],
            check=True,
        )

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)

        for session_type in cls.RUN_DESIGN:
            for subject in cls.SUBJECTS:
                n_sessions = cls.RUN_DESIGN[session_type]["n_sessions"][subject]
                for session in range(1, n_sessions + 1):
                    session_to_run = cls.RUN_DESIGN[session_type]["n_runs"][subject]
                    if session in session_to_run:
                        n_runs = session_to_run[session]
                    else:
                        n_runs = session_to_run["default"]
                    for run in range(1, n_runs + 1):
                        yield cls(
                            subject=str(subject),
                            session_type=session_type,
                            session=session,
                            run=run,
                            path=path,
                        )

    def _load_events(self) -> pd.DataFrame:
        fmri = {
            "filepath": f"method:_load_raw?timeline={self.timeline}",
            "type": "Fmri",
            "start": 0.0,
            "frequency": self._get_fmri_frequency(),
            "duration": self._get_bold_image().shape[-1] * self.TR_FMRI_S,
            "session_type": self.session_type,
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
            ses_suffix=self.session_type,
        )

        bids_events_df = read_bids_events(bids_events_df_fp, dtype={"stimulus_id": str})

        ns_events_df = self._get_img_events_df(bids_events_df, self._get_fmri_frequency())
        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: str) -> nibabel.Nifti1Image:
        return get_masked_bold_image(self._get_bold_image(), self._get_bold_mask())

    def _get_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        frequency: float,
    ) -> pd.DataFrame:
        # Keep only Stimulus presentation and Repetition blocks
        bids_events = bids_events_df[bids_events_df.event_type.isin((1, 2))].to_dict(
            "records"
        )

        id_to_name = _read_tsv_to_dict(
            Path(self.path) / self.STIMULI_FOLDER / f"{self.session_type}.tsv"
        )

        ns_events = []
        for bids_event in bids_events:
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                frequency=frequency,
                filepath=self._get_stimulus_path(id_to_name, bids_event["stimulus_id"]),
                split=(
                    "test"
                    if self.session_type != "perceptionNaturalImageTraining"
                    else "train"
                ),
                session_type=self.session_type,
            )
            ns_events.append(ns_event)

        ns_events_df = pd.DataFrame(ns_events)
        return ns_events_df

    def _get_stimulus_path(self, id_to_name: tp.Dict[str, str], stimulus_id: str) -> Path:
        if self.session_type.startswith("perceptionNaturalImage"):
            ext = ".JPEG"
            stimulus_id = stimulus_id + "0" * (15 - len(stimulus_id) - 1)
            pre, pos = stimulus_id.split(".")
            pre = "0" + pre if len(pre) == 7 else pre
            pos = pos.lstrip("0")
            stimulus_id = f"n{pre}_{pos}"
            return (
                Path(self.path)
                / self.STIMULI_FOLDER
                / self.session_type
                / f"{stimulus_id}{ext}"
            )
        elif self.session_type == "perceptionArtificialImage":
            ext = ".tiff"
        elif self.session_type == "perceptionLetterImage":
            ext = ".tif"
        else:
            raise ValueError(f"Session type {self.session_type} is incorrect")
        return (
            Path(self.path)
            / self.STIMULI_FOLDER
            / self.session_type
            / f"{id_to_name[stimulus_id]}{ext}"
        )

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
            ses_suffix=self.session_type,
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
            ses_suffix=self.session_type,
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S
