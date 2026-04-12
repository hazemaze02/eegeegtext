# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""

import re
import typing as tp
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuralset.data import BaseData

from ..download import Osf
from ..utils import success_writer


class Gifford2021(BaseData):
    subject: str
    session: int
    split: tp.Literal["train", "test"]

    # study/class level
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://osf.io/3jk45/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{gifford2022large,
        title={A large and rich EEG dataset for modeling human visual object recognition},
        author={Gifford, Alessandro T and Dwivedi, Kshitij and Roig, Gemma and Cichy, Radoslaw M},
        journal={NeuroImage},
        volume={264},
        pages={119754},
        year={2022},
        publisher={Elsevier}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.17605/OSF.IO/3JK45"
    licence: tp.ClassVar[str] = "CC BY 4.0"
    description: tp.ClassVar[str] = "10 subjects watching still images in EEG."
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack>=0.3",
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
    )

    @staticmethod
    def _create_raw_from_npy(fname: str | Path) -> mne.io.RawArray:
        """Create mne Raw object from custom npy format used by the authors."""
        out = np.load(fname, allow_pickle=True).item()

        ch_names = out["ch_names"]
        ch_names[ch_names.index("stim")] = (
            "STI101"  # Use different channel name from channel type
        )
        info = mne.create_info(ch_names, sfreq=out["sfreq"], ch_types=out["ch_types"])
        with info._unlock():
            info["lowpass"] = out["lowpass"]
            info["highpass"] = out["highpass"]
        info.set_montage("standard_1020")

        raw = mne.io.RawArray(out["raw_eeg_data"], info)
        return raw

    @classmethod
    def _download(cls, path: Path) -> None:
        # Download EEG data from OSF
        Osf(
            study="crxs4", dset_dir=path, folder="download", storage_inds=[0, 2]
        ).download()
        # storage_ind=0 is raw data (figshare storage) and 2 is description (from OSF storage)

        # Unzip EEG data
        from pyunpack import Archive

        dl_dir = path / "download"
        for zip_file in dl_dir.glob("sub-*.zip"):
            with success_writer(zip_file) as already_done:
                if not already_done:
                    Archive(str(zip_file)).extractall(str(dl_dir))

        # Load .npy and save as mne.io.Raw to enable memmaping
        for eeg_fname in dl_dir.glob("**/**/raw_eeg_*.npy"):
            with success_writer(eeg_fname) as already_done:
                if not already_done:
                    raw = cls._create_raw_from_npy(eeg_fname)
                    raw.save(str(eeg_fname).replace(".npy", "_raw.fif"), overwrite=True)

        # Download images from OSF
        Osf(study="y63gw", dset_dir=path, folder="download").download()

        # Unzip images
        for image_fname in ["training_images.zip", "test_images.zip"]:
            zip_file = dl_dir / image_fname
            with success_writer(zip_file) as already_done:
                if not already_done:
                    Archive(str(zip_file)).extractall(str(dl_dir))

        # Clean up zip files
        for zip_file in dl_dir.glob("*.zip"):
            zip_file.unlink()

    @staticmethod
    def _get_fname(path: str | Path, subject: str, session: int, split: str):
        basename = {"train": "raw_eeg_training_raw.fif", "test": "raw_eeg_test_raw.fif"}[
            split
        ]
        return (
            Path(path)
            / "download"
            / f"sub-{int(subject):02}"
            / f"ses-{session:02}"
            / basename
        )

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        """Returns a generator of all recordings"""
        for subject, session, split in product(
            range(1, 11), range(1, 5), ["train", "test"]
        ):
            fname = cls._get_fname(path, str(subject), session, split)  # type: ignore[arg-type]
            assert fname.exists(), f"File {fname} does not exist."
            yield cls(subject=str(subject), session=session, split=split, path=path)  # type: ignore[arg-type]

    def _load_events(self) -> pd.DataFrame:
        # Load image metadata to get mapping from stim codes to image filenames
        image_metadata = np.load(
            Path(self.path) / "download" / "image_metadata.npy", allow_pickle=True
        ).item()
        concept_key = str(self.split) + "_img_concepts"
        files_key = str(self.split) + "_img_files"
        event_desc = {
            i + 1: concept + "/" + basename
            for i, (concept, basename) in enumerate(
                zip(image_metadata[concept_key], image_metadata[files_key])
            )
        }

        # Extract annotations from stim channel
        raw_fname = self._get_fname(self.path, self.subject, self.session, self.split)
        raw = mne.io.read_raw(raw_fname)

        mne_events = mne.find_events(raw, stim_channel="STI101")
        annot_from_events = mne.annotations_from_events(
            events=mne_events,
            event_desc=event_desc,
            sfreq=raw.info["sfreq"],
            orig_time=raw.info["meas_date"],
        )

        # Build events dataframe
        events = pd.DataFrame(annot_from_events)
        events["description"] = events.description.apply(str)  # numpy.str_ -> str
        events["start"] = events.onset
        events["duration"] = 0.1
        events["type"] = "Image"
        image_folder = {"train": "training_images", "test": "test_images"}[self.split]  # type: ignore
        events["filepath"] = (
            str(Path(self.path) / "download" / image_folder) + "/" + events.description
        )
        events["stem"] = events.description.apply(lambda x: Path(x).stem)
        events["category"] = events["stem"].apply(lambda x: "_".join(x.split("_")[:-1]))
        events["caption"] = events.category.str.replace("_", " ").apply(
            lambda s: re.sub(r"\d", "", s)
        )
        events["session"] = self.session
        events["split"] = self.split
        events["is_test_category"] = (
            self.split == "test"
        )  # For compatibility with other THINGS

        # Add shared THINGS filepaths to images
        shared_things_path = (Path(self.path) / ".." / "THINGS-images").resolve(
            strict=False
        )
        if shared_things_path.exists():
            events["shared_filepath"] = (
                str(shared_things_path)
                + "/"
                + events.category
                + "/"
                + events.stem
                + ".jpg"
            )

        # add raw event from method
        eeg = {"filepath": raw_fname, "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])
        return events
