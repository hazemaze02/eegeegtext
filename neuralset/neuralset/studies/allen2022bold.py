# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

from neuralset.data import BaseData
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


def get_allen2022_common_path(path: str | Path) -> Path:
    path = Path(path)
    return (path / ".." / "nsd_common").resolve(strict=False)


# Helper to load 'nsd_expdesign.mat' file Copy-pasted,
# from https://github.com/ozcelikfu/brain-diffuser/blob/main/data/prepare_nsddata.py
# Commit 1c07200
def _loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    import scipy.io as spio

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class Allen2022Bold(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{allen2022massive,
        title={A massive 7T fMRI dataset to bridge cognitive neuroscience and
        artificial intelligence},
        author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L
        and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad
        and Pestilli, Franco and Charest, Ian and others},
        journal={Nature neuroscience},
        volume={25},
        number={1},
        pages={116--126},
        year={2022},
        publisher={Nature Publishing Group US New York}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.1038/s41593-021-00962-x"
    licence: tp.ClassVar[str] = "unspecified"
    description: tp.ClassVar[str] = (
        "Pre-processed timeseries fMRI data for 8 subjects"
        "watching still images in 7T fMRI"
    )

    # FIXME: add requirements
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "scipy>=1.11.4",
        "h5py>=3.10.0",
        "requests>=2.31.0",
    )

    SESSIONS_PER_SUBJECT: tp.ClassVar[tp.Dict[int, int]] = {
        1: 40,
        2: 40,
        3: 32,
        4: 30,
        5: 40,
        6: 32,
        7: 40,
        8: 30,
    }

    RUNS_PER_SESSION: tp.ClassVar[int] = 12

    N_STIMULI: tp.ClassVar[int] = 73000

    CAPTION_SEPARATOR: tp.ClassVar[str] = "\n"

    BIDS_FOLDER: tp.ClassVar[str] = "nsddata_rawdata"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives"

    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"

    TASK: tp.ClassVar[str] = "nsdcore"

    SESSION_SUFFIX: tp.ClassVar[str] = "nsd"

    TR_FMRI_S: tp.ClassVar[float] = 1.6

    session: int
    run: int | None

    @classmethod
    def _download(cls, path: Path, s3_profile: str = "saml") -> None:
        # Due to unexpected interruptions,
        # this method may have to be run more than once for retrieving all the data
        # We thus use 'aws s3 sync' instead of 'aws s3 copy' to limit the total amount
        # of data transfered when having to run 'download' multiple times
        nsd_common_path = get_allen2022_common_path(path)
        cls._download_nsd_raw_dataset(path, s3_profile)
        cls._validate_downloaded_and_fmriprepped_dataset(path)
        cls._prepare_dataset(nsd_common_path)
        cls._validate_prepared_dataset(nsd_common_path)

    @classmethod
    def _validate_downloaded_and_fmriprepped_dataset(cls, path: Path) -> None:
        nsd_common_path = get_allen2022_common_path(path)
        for case_ in cls._iter_subject_session_run():
            subject, session, run = case_
            # BOLD image exists
            fps = []
            fps.append(
                get_bids_filepath(
                    path / cls.DERIVATIVES_FOLDER,
                    subject=subject,
                    session=session,
                    run=run,
                    task=cls.TASK,
                    filetype="bold",
                    data_type="Fmri",
                    space=cls.BOLD_SPACE,  # type: ignore
                    ses_suffix=cls.SESSION_SUFFIX,
                )
            )
            # BOLD image mask exists
            fps.append(
                get_bids_filepath(
                    path / cls.DERIVATIVES_FOLDER,
                    subject=subject,
                    session=session,
                    run=run,
                    task=cls.TASK,
                    filetype="bold_mask",
                    data_type="Fmri",
                    space=cls.BOLD_SPACE,  # type: ignore
                    ses_suffix=cls.SESSION_SUFFIX,
                )
            )
            # events file exists
            fps.append(
                get_bids_filepath(
                    path / cls.BIDS_FOLDER,
                    subject=subject,
                    session=session,
                    run=run,
                    task=cls.TASK,
                    filetype="events",
                    data_type="Fmri",
                    ses_suffix=cls.SESSION_SUFFIX,
                )
            )
            fps.extend(
                [
                    nsd_common_path / filename
                    for filename in [
                        "nsd_expdesign.mat",
                        "nsd_stimuli.hdf5",
                        "COCO_73k_annots_curated.npy",
                    ]
                ]
            )
            for fp in fps:
                if not fp.exists():
                    raise RuntimeError(f"Missing file {fp} for case {case_}")

    @classmethod
    def _validate_prepared_dataset(cls, path: Path):
        cls._validate_file_count(path, "nsd_captions", ".npy")
        cls._validate_file_count(path, "nsd_stimuli", ".png")

    @classmethod
    def _validate_file_count(cls, path: Path, sub_dir: str, extension: str):
        files = [
            filename
            for filename in (path / sub_dir).iterdir()
            if filename.suffix == extension
        ]

        assert len(files) == cls.N_STIMULI, (
            f"There should be {cls.N_STIMULI} {extension} files in"
            f" {path / sub_dir} but found only {len(files)}"
        )

    @classmethod
    def _download_nsd_raw_dataset(cls, path: Path, s3_profile: str) -> None:
        path.mkdir(exist_ok=True, parents=True)

        nsd_common_path = get_allen2022_common_path(path)
        path.mkdir(exist_ok=True, parents=True)

        aws_cmds = []

        # Raw BOLD fMRI
        raw_bold_aws_cmd = (
            f"aws s3 --profile {s3_profile} sync"
            " s3://natural-scenes-dataset/nsddata_rawdata/"
            f" {path}/{cls.BIDS_FOLDER}"
        )
        aws_cmds.append(raw_bold_aws_cmd)

        # Experimental design matrix (downloaded to nsd common folder)
        expdesign_mat_aws_cmd = (
            f"aws s3 --profile {s3_profile} cp"
            " s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat"
            f" {nsd_common_path}"
        )
        aws_cmds.append(expdesign_mat_aws_cmd)

        # Stimulus matrix (downloaded to nsd common folder)
        stimuli_mat_aws_cmd = (
            f"aws s3 --profile {s3_profile} cp"
            " s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
            f" {nsd_common_path}"
        )
        aws_cmds.append(stimuli_mat_aws_cmd)

        for aws_cmd in aws_cmds:
            subprocess.run(aws_cmd, shell=True)

        # Image captions (downloaded to nsd common folder)
        url = (
            "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/"
            "COCO_73k_annots_curated.npy"
        )
        import requests

        response = requests.get(url)
        (nsd_common_path / "COCO_73k_annots_curated.npy").write_bytes(response.content)

    @classmethod
    def _prepare_dataset(cls, path: Path) -> None:
        cls._extract_stimuli(path)
        cls._extract_captions(path)
        cls._extract_test_images_ids(path)

    @classmethod
    def _extract_stimuli(cls, path: Path) -> None:
        # Image stimuli are stored in an hdf5 file
        import h5py

        f_stim = h5py.File(path / "nsd_stimuli.hdf5", "r")
        stim = f_stim["imgBrick"][:]

        nsd_stimuli_folder = path / "nsd_stimuli"
        nsd_stimuli_folder.mkdir(exist_ok=True, parents=True)

        for idx in trange(stim.shape[0]):
            Image.fromarray(stim[idx]).save(nsd_stimuli_folder / f"{idx}.png")

    @classmethod
    def _extract_captions(cls, path: Path) -> None:
        path_to_caption_npys = path / "nsd_captions"
        path_to_caption_npys.mkdir(exist_ok=True, parents=True)

        # Human curated version of NSD captions
        path_to_caption_file = path / "COCO_73k_annots_curated.npy"
        captions = np.load(path_to_caption_file, mmap_mode="r")

        for idx in trange(captions.shape[0]):
            annots_idx = np.array(
                [annot for annot in captions[idx] if len(annot.strip()) > 0]
            )
            np.save(path_to_caption_npys / f"{idx}.npy", annots_idx)

    @classmethod
    def _extract_test_images_ids(cls, path: Path) -> None:
        path_to_expdesign_mat = path / "nsd_expdesign.mat"
        expdesign_mat = _loadmat(path_to_expdesign_mat)
        np.save(
            path / "test_images_ids.npy",
            expdesign_mat["sharedix"],
        )

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)
        cls._validate_downloaded_and_fmriprepped_dataset(path)

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
            run=self.run,  # type: ignore
            task=self.TASK,
            filetype="events",
            data_type="Fmri",
            ses_suffix=self.SESSION_SUFFIX,
        )

        bids_events_df = read_bids_events(bids_events_df_fp)
        path_to_stimuli = get_allen2022_common_path(self.path) / "nsd_stimuli"
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df,
            path_to_stimuli,
        )
        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: str) -> nibabel.Nifti1Image:
        return get_masked_bold_image(self._get_bold_image(), self._get_bold_mask())

    def _get_test_image_ids(self) -> np.ndarray:
        return np.load(
            get_allen2022_common_path(self.path) / "test_images_ids.npy"
        ).tolist()

    def _get_captions(self, image_id: int) -> str:
        assert (
            image_id >= 0
        ), f"Parameter 'image_id' has value {image_id} but should be positive"

        # captions per image are saved on disk with a 0-based index (0 to 72999)
        captions = np.load(
            get_allen2022_common_path(self.path) / f"nsd_captions/{image_id}.npy"
        ).tolist()
        captions = [cap.replace(self.CAPTION_SEPARATOR, "") for cap in captions]
        return self.CAPTION_SEPARATOR.join(captions)

    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        stimuli_path: str | Path,
    ) -> pd.DataFrame:
        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            image_id = bids_event["73k_id"]  # 1-based
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                filepath=str(Path(stimuli_path) / f"{image_id-1}.png"),
                split=("test" if image_id in self._get_test_image_ids() else "train"),
                caption=self._get_captions(image_id - 1),
            )
            ns_events.append(ns_event)

        ns_events_df = pd.DataFrame(ns_events)
        return ns_events_df

    @classmethod
    def _iter_subject_session_run(cls):
        for subject in cls.SESSIONS_PER_SUBJECT.keys():
            for session in range(1, cls.SESSIONS_PER_SUBJECT[subject] + 1):
                for run in range(1, cls.RUNS_PER_SESSION + 1):
                    yield (subject, session, run)

    def _get_bold_mask(self):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=self.session,
            run=self.run,  # type : ignore
            task=self.TASK,
            filetype="bold_mask",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            ses_suffix=self.SESSION_SUFFIX,
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
            ses_suffix=self.SESSION_SUFFIX,
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S
