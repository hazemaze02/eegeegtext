# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Corpus of EEG recordings from the PhysioNet "You Snooze You Win: The PhysioNet/Computing in Cardiology Challenge 2018".
https://physionet.org/content/challenge-2018/

Data captured following the AASM standards by the Computational Clinical Neurophysiology Laboratory and the Clinical Data Animation Laboratory at Massachusetts General Hospital (MGH).
Includes 1,983 subjects monitored at a MGH sleep laboratory for the diagnosis of sleep disorders.
Partitioned into train (N=994) and test sets (N=989).

EEG is referenced at M1 and M2 (left and right mastoid processes).
"""

import logging
import shutil
import subprocess
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from neuralset.data import BaseData

from ..utils import success_writer

logger = logging.getLogger(__name__)
logger.propagate = False


class Ghassemi2018(BaseData):
    split: tp.Literal["train", "test"]
    split_dir: tp.Literal["training", "test"]
    prefix: str
    subject: str

    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://physionet.org/files/challenge-2018/1.0.0/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{ghassemi2018,
        title={You Snooze, You Win: the PhysioNet/Computing in Cardiology Challenge 2018},
        author={Ghassemi, Mohammad M and Moody, Benjamin E and Lehman, Li-wei H and Song, Christopher and Li, Qiao, Sun, Haoqi and Mark, Roger G and Westover, M Brandon and Clifford, Gari D },
        year={2018},
        journal={Computing in Cardiology},
        volume={45},
        pages={1-4},
        publisher={Institute of Electrical and Electronics Engineers}
    }
        """
    doi: tp.ClassVar[str] = "doi:10.22489/CinC.2018.049"
    description: tp.ClassVar[str] = (
        "A corpus of 1,985 EEG recordings from a sleep laboratory annotated for arousal and sleep stage."
    )

    @classmethod
    def _download(cls, path: Path) -> None:
        #  Option to download through S3
        #  aws s3 sync s3://physionet-open/challenge-2018/1.0.0/ DESTINATION
        folder = path / "download"
        folder.mkdir(exist_ok=True, parents=True)
        for sub_d in ["training/", "test/", "RECORDS", "new-arousals.zip", "age-sex.csv"]:
            download_url = f"{cls.url}/{sub_d}"
            rename_path = folder / sub_d
            with success_writer(rename_path) as already_done:
                if not already_done:
                    subprocess.run(
                        (f"wget -r -N -c -np -P {folder} {download_url}"),
                        shell=True,
                    )
                    download_path = (
                        folder / "physionet.org/files/challenge-2018/1.0.0" / sub_d
                    )
                    download_path.rename(rename_path)
                    shutil.rmtree(f"{folder}/physionet.org")  # cleanup empty folder
                    logger.info(f"Downloaded files to {rename_path}.")

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        folder = Path(path) / "download"
        for split in ["training", "test"]:
            split_dir = folder / split
            for file_path in split_dir.rglob("*.mat"):
                # Skip arousal files in training set
                if "arousal.mat" in str(file_path):
                    continue
                prefix, subject = file_path.stem.split("-")
                yield cls(
                    prefix=prefix,
                    subject=subject,
                    path=path,
                    split=split.replace("ing", ""),  # type: ignore
                    split_dir=split,  # type: ignore
                )

    def _load_events(
        self,
    ) -> pd.DataFrame:
        # TODO: Add arousal events ("*-arousal.mat")
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=f"method:_load_raw?timeline={self.timeline}",
                    split=self.split,
                ),
            ]
        )

    def _get_eeg_filenames(self) -> tuple[Path, Path]:
        subject_path = (
            Path(self.path)
            / "download"
            / self.split_dir
            / f"{self.prefix}-{self.subject}"
        )
        eeg_file = subject_path / f"{self.prefix}-{self.subject}.mat"
        header_file = subject_path / f"{self.prefix}-{self.subject}.hea"
        return eeg_file, header_file

    def _load_raw(
        self,
        timeline: str,
        key: str = "val",
    ) -> mne.io.RawArray:
        # EEG reference mastoid sites (M1, M2)
        eeg_file, header_file = self._get_eeg_filenames()
        with open(header_file, "r") as f:
            header = f.read().split("\n")

        # Load recording file as an array.
        eeg_data = sio.loadmat(eeg_file)[key]

        s_freq = np.array(header[0].split()[1:], dtype=int)[1]
        ch_names = [ch.split()[-1] for ch in header[1:-1]]

        info = mne.create_info(ch_names=ch_names, sfreq=s_freq)

        eeg = mne.io.RawArray(eeg_data, info)

        return self._fix_ch_names(eeg)

    @staticmethod
    def _fix_ch_names(raw: mne.io.RawArray) -> mne.io.RawArray:
        # Standard channels in 10-5 montage
        montage = mne.channels.make_standard_montage("standard_1005")

        ch_types, ch_names_mapping = {}, {}
        for name in raw.ch_names:
            ch_names_mapping[name] = name.split("-")[0]
            if name == "ECG":
                ch_type = "ecg"
            elif "Chin" in name:
                ch_type = "emg"
            elif ch_names_mapping[name] in montage.ch_names:
                ch_type = "eeg"
            else:
                ch_type = "misc"
            ch_types[ch_names_mapping[name]] = ch_type

        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)
        raw = raw.set_montage(montage, on_missing="ignore")

        return raw
