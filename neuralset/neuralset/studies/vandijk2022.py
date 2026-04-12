# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Two Decades Brainclinics Research Archive for Insights in Neurophysiology (TDBRAIN)
is a database of clinical EEG scans for 1,274 psychiatric patients. EEG was measured
in the resting state for both eyes-open and -closed states. Additional demographic
and behavioral measures are available that have not been incorporated in this script.

Amongst others, common clinical indications include:
- Major Depressive Disorder (MDD; N=426)
- Attention deficit hyperactivity disorder (ADHD; N=271)
- Subjective Memory Complaint (SMC: N=119)
- Obsessive-compulsive disorder (OCD; N=75)
"""

import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralset.data import BaseData


class Vandijk2022(BaseData):
    session: str
    task: tp.Literal["task-restEO", "task-restEC"]

    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "http://www.brainclinics.com/resources"
    bibtex: tp.ClassVar[
        str
    ] = """
        @article{Vandijk2022,
            title={The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database},
            author={Van Dijk, Hanneke and Van Wingen, Guido and Denys, Damiaan and Olbrich, Sebastian and Van Ruth, Rosalinde and Arns, Martijn},
            journal={Scientific Data},
            volume={9},
            number={1},
            pages={333},
            year={2022},
            publisher={Nature Publishing Group UK London}
        }
        """
    doi: tp.ClassVar[str] = "10.1038/s41597-022-01409-z"
    description: tp.ClassVar[str] = (
        "Lifespan database (5–89 years) containing raw rs-EEG of a heterogenous collection of 1,274 psychiatric patients collected between 2001 to 2021."
    )

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        #  Requires login with ORCID and accepting DUA
        #  Downloads a password-protected zip file
        # download_url = "https://brainclinics.com/restricted-2/"
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        folder = Path(path) / "download" / "derivatives"
        for sub_dir in folder.iterdir():
            for sess_dir in sub_dir.iterdir():
                for file_path in sess_dir.rglob("*_eeg.csv"):
                    subject, session, task, _ = file_path.stem.split("_")
                    yield cls(
                        path=path,
                        subject=subject,
                        session=session,
                        task=task,  # type: ignore
                    )

    def _get_eeg_filename(self) -> Path:
        eeg_file = (
            Path(self.path)
            / "download"
            / "derivatives"
            / self.subject
            / self.session
            / "eeg"
            / f"{self.subject}_{self.session}_{self.task}_eeg.csv"
        )
        return eeg_file

    def _load_events(self) -> pd.DataFrame:
        pt_info = self._get_participant_info()
        label = pt_info["label"]
        split = pt_info["split"]
        eye_state_dict = {"task-restEO": "open", "task-restEC": "closed"}
        return pd.concat(
            [
                pd.DataFrame(
                    [
                        dict(
                            type="Eeg",
                            start=0.0,
                            filepath=f"method:_load_raw?timeline={self.timeline}",
                            label=label,
                            split=split,
                        ),
                    ]
                ),
                pd.DataFrame(
                    [
                        dict(
                            type="EyeState",
                            start=0,
                            state=eye_state_dict[self.task],
                            label=label,
                            split=split,
                        ),
                    ]
                ),
            ]
        )

    def _get_participant_info(self):
        # load
        file_name = f"{self.path}/download/TDBRAIN_participants_V2.tsv"
        info_df = pd.read_csv(file_name, sep="\t")
        info_df = info_df.set_index("participants_ID")
        info_df["indication"] = info_df["indication"].replace({"REPLICATION": ""})
        info_df["DISC/REP"] = info_df["DISC/REP"].replace(
            {"REPLICATION": "training", "DISCOVERY": "test"}
        )
        # extract participant info
        pt_info = {}
        col_dict = {"indication": "label", "DISC/REP": "split"}
        for col, info in col_dict.items():
            try:
                pt_info[info] = info_df.loc[self.subject, col].lower()
            except:
                # handle missing info
                pt_info[info] = ""
        return pt_info

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # 26 channels, 10–10 electrode international system
        raw = pd.read_csv(self._get_eeg_filename())
        ch_names = list(raw.columns)
        s_freq = 500.0
        info = mne.create_info(ch_names=ch_names, sfreq=s_freq)
        eeg = mne.io.RawArray(raw.T, info)
        return self._fix_ch_info(eeg)

    @staticmethod
    def _fix_ch_info(eeg: mne.io.RawArray) -> mne.io.RawArray:
        # Standard channels in 10-5 montage
        montage = mne.channels.make_standard_montage("standard_1005")

        ch_types = {}
        eyetrack_channels = ["HNHR", "HPHL", "VNVB", "VPVA", "OrbOcc"]
        for name in eeg.ch_names:
            if name == "Erbs":  # cervical bone
                ch_type = "ecg"
            elif name == "Mass":  # right masseter muscle
                ch_type = "emg"
            elif name in montage.ch_names:
                ch_type = "eeg"
            elif name in eyetrack_channels:
                # Dropped
                continue
            else:
                ch_type = "misc"
            ch_types[name] = ch_type

        eeg = eeg.drop_channels(eyetrack_channels)
        eeg = eeg.set_channel_types(ch_types)
        eeg = eeg.set_montage(montage, on_missing="ignore")

        return eeg
