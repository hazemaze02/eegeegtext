# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Corpus of 1,526 EEG recordings from the PhysioNet "EEG Motor Movement/Imagery Dataset".
https://www.physionet.org/content/eegmmidb/

Experimental runs
1. Baseline, eyes open
2. Baseline, eyes closed
3. Task 1 (open and close left or right fist)
4. Task 2 (imagine opening and closing left or right fist)
5. Task 3 (open and close both fists or both feet)
6. Task 4 (imagine opening and closing both fists or both feet)
7. Task 1
8. Task 2
9. Task 3
10. Task 4
11. Task 1
12. Task 2
13. Task 3
14. Task 4
"""

import logging
import shutil
import subprocess
import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralset.data import BaseData

from ..utils import success_writer

logger = logging.getLogger(__name__)
logger.propagate = False


class Schalk2009(BaseData):
    run: str

    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://www.physionet.org/content/eegmmidb/1.0.0/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{schalk2009,
        author={Schalk, G. and McFarland, D.J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J.R.},
        journal={IEEE Transactions on Biomedical Engineering},
        title={BCI2000: a general-purpose brain-computer interface (BCI) system},
        year={2004},
        volume={51},
        number={6},
        pages={1034-1043},
    }
    """
    doi: tp.ClassVar[str] = "doi:10.22489/CinC.2018.049"
    description: tp.ClassVar[str] = (
        "Corpus of 1,526 EEG recordings (1-2 mins) from 109 participants performing motor and imagery tasks."
    )

    @classmethod
    def _download(cls, path: Path) -> None:
        #  Option to download through S3
        #  aws s3 sync s3://physionet-open/eegmmidb/1.0.0/ DESTINATION
        folder = path / "download"
        download_url = "https://physionet.org/files/eegmmidb/1.0.0/"
        with success_writer(folder) as already_done:
            if not already_done:
                folder.mkdir(exist_ok=True, parents=True)
                subprocess.run(
                    (f"wget -r -N -c -np -P {folder} {download_url}"),
                    shell=True,
                )
                download_path = folder / "physionet.org/files/eegmmidb/1.0.0"
                download_path.rename(folder)
                shutil.rmtree(f"{folder}/physionet.org")  # cleanup empty folder
                logger.info(f"Downloaded files to {folder}.")

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        folder = Path(path) / "download"
        for sub_dir in folder.iterdir():
            for file_path in sub_dir.rglob("*.edf"):
                subject = file_path.stem[:4]
                run = file_path.stem[4:]
                yield cls(
                    path=path,
                    subject=subject,
                    run=run,
                )

    def _load_events(self) -> pd.DataFrame:
        eeg_events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    filepath=f"method:_load_raw?timeline={self.timeline}",
                    start=0.0,
                ),
            ]
        )
        motor_events = self._get_event_annotations()
        return pd.concat([eeg_events, motor_events]).reset_index()

    def _get_eeg_filename(self) -> Path:
        return (
            Path(self.path) / "download" / self.subject / f"{self.subject}{self.run}.edf"
        )

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        raw = mne.io.read_raw_edf(self._get_eeg_filename())
        replacements = {
            ".": "",
            "Af": "AF",
            "Cp": "CP",
            "Fc": "FC",
            "Po": "PO",
            "Tp": "TP",
            "Ft": "FT",
        }
        ch_types, ch_names_mapping = {}, {}
        for ch in raw.ch_names:
            fixed_ch = ch
            for key in replacements.keys():
                fixed_ch = fixed_ch.replace(key, replacements[key])
            ch_names_mapping[ch] = fixed_ch
            ch_types[fixed_ch] = "eeg"
        montage = mne.channels.make_standard_montage("standard_1005")
        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)
        raw = raw.set_montage(montage, on_missing="ignore")
        to_drop = [
            name
            for name in raw.ch_names
            if ch_types[name] == "eeg" and name not in montage.ch_names
        ]
        raw = raw.drop_channels(to_drop)
        if len(to_drop) > 0:
            logger.info("Dropped %s unrecognized EEG channels: %s", len(to_drop), to_drop)
        raw = raw.set_montage(montage, on_missing="ignore")
        return raw

    def _get_event_annotations(self):
        eeg = mne.io.read_raw_edf(self._get_eeg_filename())
        motor_df = pd.DataFrame(
            columns=[
                "type",
                "filepath",
                "start",
                "duration",
                "frequency",
                "task",
                "code",
                "description",
                "state",
            ]
        )

        DESC_CODE_MAPPING = {
            "rest": 0,
            "motor_left_fist": 1,
            "motor_right_fist": 2,
            "motor_bilateral_fist": 3,
            "motor_bilateral_feet": 4,
            "imagery_left_fist": 5,
            "imagery_right_fist": 6,
            "imagery_bilateral_fist": 7,
            "imagery_bilateral_feet": 8,
        }
        annots = eeg.annotations

        # Rest runs
        if self.run in ["R01", "R02"]:
            event_type = "EyeState"
            task = "Rest"
            # Skip annotation for unknown task
            if set(annots.description) != {"T0"}:
                return motor_df
            if self.run == "R01":
                description_dict = {"T0": "open"}
            else:
                description_dict = {"T0": "closed"}
        # Motor runs
        elif self.run in ["R03", "R07", "R11", "R05", "R09", "R13"]:
            event_type = "Stimulus"
            task = "Motor"
            if self.run in ["R03", "R07", "R11"]:
                description_dict = {
                    "T0": "rest",
                    "T1": "motor_left_fist",
                    "T2": "motor_right_fist",
                }
            else:
                description_dict = {
                    "T0": "rest",
                    "T1": "motor_bilateral_fist",
                    "T2": "motor_bilateral_feet",
                }
        # Motor imagery runs
        elif self.run in ["R04", "R08", "R12", "R06", "R10", "R14"]:
            event_type = "Stimulus"
            task = "Imagery"
            if self.run in ["R04", "R08", "R12"]:
                description_dict = {
                    "T0": "rest",
                    "T1": "imagery_left_fist",
                    "T2": "imagery_right_fist",
                }
            else:
                description_dict = {
                    "T0": "rest",
                    "T1": "imagery_bilateral_fist",
                    "T2": "imagery_bilateral_feet",
                }
        else:
            ValueError(
                f"Invalid input for variable. Got run={self.run}, expected R[00-12]."
            )

        motor_df["start"] = annots.onset
        motor_df["duration"] = annots.duration
        motor_df["frequency"] = eeg.info["sfreq"]
        motor_df["type"] = event_type
        motor_df["task"] = task
        if event_type == "EyeState":
            motor_df["state"] = [description_dict[c] for c in annots.description]
        else:
            motor_df["description"] = [description_dict[c] for c in annots.description]
            motor_df["code"] = [DESC_CODE_MAPPING[c] for c in motor_df["description"]]

        return motor_df
