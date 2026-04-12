# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lemon Dataset.

- Subject 15 and 100 have shorter recordings (6m29s and 13m53s respectively) than others.
- Subject 20, 44, 193 and 219 have wrong path in the ".vhdr" file to the eeg file and marker
  file, and we need to tweak the path after downloading them.
- Subject 59 and 81 have some dummy "eye closed" events in the beginning of the recordings,
  and we need to have customized code logic to handle them.
- Subject 78 and 126 use "Stimulus/S208" instead of "Stimulus/S210" to represent "eye closed"
  , and we have to use customized code block to handle them.
- Subject 203 does not have ".vmrk" marker file and thus we cannot fetch "EyeState" events.
- Subject 235, 237, 259, 281 and 293 are missing from the dataset.
- Subject 285 does not have any "EyeState" events in the recording.

"""

import os
import re
import typing as tp
import urllib
from pathlib import Path

import mne
import pandas as pd
import requests

from neuralset.data import BaseData


class Babayan2019(BaseData):
    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = (
        "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{babayan2019amind-brain-body,
        title={A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults.},
        author={Babayan, A., Erbey, M., Kumral, D. et al.},
        volume={6},
        number={1},
        pages={180308},
        year={2019},
        journal={Scientific Data},
        publisher={Nature Publishing Group UK London}
    }
        """
    doi: tp.ClassVar[str] = "doi:10.1038/sdata.2018.308"
    description: tp.ClassVar[str] = (
        "227 healthy subjects resting with eye closed / eye open in EEG."
    )
    licence: tp.ClassVar[str] = "CC BY 4.0 DEED"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("bs4",)

    # Constants
    # The majority of subjects have "Stimulus/210" to represent "eye closed" as pointed in the paper.
    # However, two subjects use "Stimulus/S208" (partly or fully) instead of "Stimulus/S210".
    # This mapping takes care of that.
    stim_mapping: tp.ClassVar[tp.Dict[str, str]] = {
        "Stimulus/S200": "open",
        "Stimulus/S210": "closed",
        "Stimulus/S208": "closed",
    }

    @classmethod
    def _download(cls, path: Path) -> None:
        from bs4 import BeautifulSoup

        base_url = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
        page = requests.get(base_url).text
        soup = BeautifulSoup(page, "html.parser")
        subjects = [
            node.get("href").strip("/")
            for node in soup.find_all("a")
            if node.get("href").startswith("sub")
        ]

        out_base = Path(path) / "lemon"
        if not out_base.exists():
            out_base.mkdir(parents=True, exist_ok=True)

        extensions = ["eeg", "vhdr", "vmrk"]

        def _download_single_file(sub: str, ext: str) -> None:
            """Download a file with a given extension for a given subject."""
            sub_url = f"{sub}/RSEEG/{sub}.{ext}"
            url = f"{base_url}/{sub_url}"
            out_path = out_base / sub / "RSEEG"
            if not out_path.exists():
                os.makedirs(out_path)
            out_name = out_path / f"{sub}.{ext}"
            try:
                urllib.request.urlretrieve(url, out_name)
            except Exception as err:
                print(err)

        for sub in subjects:
            for ext in extensions:
                _download_single_file(sub, ext)

        cls._fix_vhdr(out_base)

    @classmethod
    def _fix_vhdr(cls, path: Path) -> None:
        """Check data and marker file paths in ".vhdr" file, and fix them if needed.

        A valid ".vhdr" file should have "DataFile" point to the correct EEG file, and "MarkerFile" to
        the right ".vmrk" file. For example, for subject "010002":

        "sub-010002.vhdr":
        ...
        [Common Infos]
        Codepage=UTF-8
        DataFile=sub-010002.eeg
        MarkerFile=sub-010002.vmrk
        ...


        """
        for sub in path.iterdir():
            # subject examples: "010002", "010005", ...
            subject = sub.name.removeprefix("sub-")
            vhdr_path = sub / "RSEEG" / f"sub-{subject}.vhdr"
            if not vhdr_path.exists():
                continue

            pattern = r"\nDataFile=(.+)\.eeg\nMarkerFile=(.+)\.vmrk\n"
            with open(vhdr_path, "r") as fp:
                content = fp.read()
                matches = re.findall(pattern=pattern, string=content)

            if (
                len(list(matches)) == 1
                and matches[0][0] == f"sub-{subject}"
                and matches[0][1] == f"sub-{subject}"
            ):
                continue

            replacement = f"\nDataFile=sub-{subject}.eeg\nMarkerFile=sub-{subject}.vmrk\n"
            with open(vhdr_path, "w") as fp:
                content = re.sub(pattern=pattern, repl=replacement, string=content)
                fp.write(content)

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ) -> tp.Iterator["Babayan2019"]:
        """Returns a generator of all recordings"""
        folder = Path(path) / "lemon"
        assert folder.exists()

        for subject in folder.iterdir():
            yield cls(subject=subject.name.removeprefix("sub-"), path=path)

    def _load_events(self) -> pd.DataFrame:
        # Read ".vhdr" file and mne will look for other files (".eeg", ".vmrk").
        file_path = (
            Path(self.path)
            / "lemon"
            / f"sub-{self.subject}"
            / "RSEEG"
            / f"sub-{self.subject}.vhdr"
        )

        # Eye Status
        df = self._get_eye_state_events(file_path)

        # EEG event
        eeg = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=file_path,
                ),
            ]
        )
        df = pd.concat([eeg, df])
        return df

    def _get_eye_state_events(self, file_path: Path) -> pd.DataFrame:
        """Fetch EyeState events from annotations."""
        raw = mne.io.read_raw_brainvision(file_path)
        df = raw.annotations.to_data_frame()

        df["state"] = df["description"].replace(self.stim_mapping)

        # Compute time offset
        df["onset"] = pd.to_datetime(df["onset"])
        df["start"] = (df["onset"] - df["onset"].iloc[0]).dt.total_seconds()

        # Subject 59 and 81 have small "closed" blocks at the beginning, each block
        # starting with "New Segment/". Remove those blocks and start from the last
        # segment.
        last_new_segment = df.loc[df["description"] == "New Segment/"].index[-1]
        if last_new_segment < 20:
            # Subject 232 has the last "New Segment/" at index 65 which does not indicate a new segment
            # (false positive). All other subjects have the last "New Segment/" at index <= 16.
            df = df.loc[last_new_segment + 1 :]

        df = df[df["state"].isin(["open", "closed"])]

        # Keep only the first trigger for each "open" and "closed" block.
        df = df[df["state"] != df["state"].shift(1)]

        df = df.drop(columns=["onset", "description"])
        df.reset_index(drop=True, inplace=True)
        df["type"] = "EyeState"
        df["duration"] = 60.0  # 1 min for each "open" and "closed" block.
        return df
