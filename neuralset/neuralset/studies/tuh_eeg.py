# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The TUH-EEG dataset is a large corpus of clinical EEG recordings from Temple University Hospital.

Subsets of these recordings have various labels or annotations, described as follows:
- Obeid2016: "tueg"
  - Superset of TUH-EEG with EEG for all subjects without labels or annotations
- Lopez2017: "tuab"
  - Subset of TUH-EEG with labels for normal / abnormal recordings
- Hamid2020: "tuar"
  - Subset of TUH-EEG with artifact events
- Veloso2017: "tuep"
  - Subset of TUH-EEG with labels no epilepsy / epilepsy recordings
- Harati2015: "tuev"
  -  Subset of TUH-EEG with annotations for epilepsy and artifact events
- VonWeltin2017: "tusl"
  - Subset of TUH-EEG with annotations for slowing events
- Shah2018: "tusz"
  - Subset of TUH-EEG with extensive annotations for seizure and artifact events
"""


import datetime
import logging
import os
import re
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuralset.data import BaseData

logger = logging.getLogger(__name__)


def scantree(path: str | Path) -> tp.Iterator[str]:
    """Recursively yield filepaths from given directory.

    Note
    ----
    About 2x faster than a combination of Path.iterdir and Path.rglob for walking through
    Obeid2016's directory.
    """
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry.path


class _BaseTuhEeg(BaseData):
    split: tp.Literal["train", "eval", "dev"] | None = None
    label: (
        tp.Literal[
            "abnormal",
            "normal",
            "epilepsy",
            "no_epilepsy",
            "bckg",  # Background (no seizure)
            "gped",  # Generalized periodic epileptiform discharges
            "pled",  # Periodic lateralized epileptiform discharges
            "spsw",  # Spike and/or sharp waves
        ]
        | None
    ) = None

    STUDY_CODE_MAP: tp.ClassVar[dict] = {
        "obeid2016": "tueg",
        "lopez2017": "tuab",
        "hamid2020": "tuar",
        "veloso2017": "tuep",
        "harati2015": "tuev",
        "vonweltin2017": "tusl",
        "shah2018": "tusz",
    }
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
    description: tp.ClassVar[str] = (
        "The TUH-EEG dataset is a large corpus of clinical EEG recordings from Temple University Hospital."
    )

    # TODO: Add download method, requires authentication
    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError("Dataset not available to download yet.")
        # cls._create_symbolic_links(path)

    @classmethod
    def _create_symbolic_links(cls, path: str | Path) -> None:
        """Makes symbolic link for each tuh_eeg study folder"""
        for study, code in cls.STUDY_CODE_MAP.items():
            source_path = Path(path) / "tuh_eeg" / code
            target_path = Path(path) / study
            target_path.symlink_to(source_path)

    @staticmethod
    def _fix_ch_names(raw: mne.io.RawArray) -> mne.io.RawArray:
        # XXX Some electrodes, e.g. T1, T2 are not part of this montage; find a way to include them
        pattern = re.compile("EEG (.*?)-(REF|LE)")
        ch_types, ch_names_mapping = {}, {}
        for name in raw.ch_names:
            # Clean up name
            match = pattern.match(name)
            ch_names_mapping[name] = (
                name if match is None else match[1].replace("FP", "Fp").replace("Z", "z")
            )
            # Infer channel type
            if "EKG" in name:
                ch_type = "ecg"
            elif "EMG" in name:
                ch_type = "emg"
            elif name.startswith("EEG "):
                ch_type = "eeg"
            else:
                ch_type = "misc"
            ch_types[ch_names_mapping[name]] = ch_type

        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)

        # Drop EEG channels that were not found in the 10-5 montage
        montage = mne.channels.make_standard_montage("standard_1005")
        to_drop = sorted(
            [
                name
                for name in raw.ch_names
                if ch_types[name] == "eeg" and name not in montage.ch_names
            ]
        )
        raw = raw.drop_channels(to_drop)
        if len(to_drop) > 0:
            logger.info("Dropped %s unrecognized EEG channels: %s", len(to_drop), to_drop)
        raw = raw.set_montage(montage, on_missing="ignore")

        return raw

    def _load_raw_from_path(self, file_path: str) -> mne.io.RawArray:
        raw = mne.io.read_raw(file_path)
        raw = self._fix_ch_names(raw)
        # Some files have an invalid measurement date; replacing by single date as we don't use
        # this information anyway
        raw.info.set_meas_date(
            datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        )
        return raw

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        eeg_file = self._get_eeg_filename()
        raw = self._load_raw_from_path(eeg_file)
        return raw

    def _get_eeg_filename(self):
        pass

    def _load_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=f"method:_load_raw?timeline={self.timeline}",
                    split=self.split,
                    label=self.label,
                )
            ]
        ).dropna(axis=1)


class Obeid2016(_BaseTuhEeg):
    token_number: str  # E.g. "t000"
    session: str  # E.g. "s001"
    channel_configuration: tp.Literal[
        "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
    ]  # AR reference configuration.
    date: str  # YYYY or YYYY_MM_DD, E.g., 2000
    folder_number: str  # E.g. "000" through "109"
    prefix: str | None

    # Class variables
    description: tp.ClassVar[str] = (
        "Superset TUH-EEG with EEG for all subjects without labels or annotations"
    )
    url: tp.ClassVar[str] = "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Obeid2016,
        title={The Temple University Hospital EEG Data Corpus},
        author={Obeid, I., & Picone, J.},
        year={2016},
        journal={Frontiers in Neuroscience},
        volume={10},
        number={196},
    }
    """
    doi: tp.ClassVar[str] = "doi:10.3389/fnins.2016.00196"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""

        folder = Path(path) / "edf"
        for fname in scantree(folder):
            if not fname.endswith(".edf"):
                continue
            folder_number, _, sess_dir, channel_configuration, file_path = Path(
                fname
            ).parts[-5:]
            date = sess_dir[5:]
            file_path = file_path.replace(".edf", "")
            if len(file_path.split("_")) == 4:
                prefix, subject, session, token_number = file_path.split("_")
            else:
                subject, session, token_number = file_path.split("_")
                prefix = None
            yield cls(
                path=str(path),
                folder_number=folder_number,
                subject=subject,
                session=session,
                date=date,
                channel_configuration=channel_configuration,  # type: ignore
                token_number=token_number,
                prefix=prefix,  # type: ignore
            )

    def _get_eeg_filename(self) -> str:
        eeg_path = (
            Path(self.path)
            / "edf"
            / self.folder_number
            / self.subject
            / f"{self.session}_{self.date}"
            / self.channel_configuration
        )
        if self.prefix is None:
            eeg_file = str(
                eeg_path / f"{self.subject}_{self.session}_{self.token_number}.edf"
            )
        else:
            eeg_file = str(
                eeg_path
                / f"{self.prefix}_{self.subject}_{self.session}_{self.token_number}.edf"
            )
        return eeg_file


class Lopez2017(_BaseTuhEeg):
    token_number: str  # E.g. "t000"
    channel_configuration: str = "01_tcp_ar"
    split: tp.Literal["train", "eval"]
    label: tp.Literal["abnormal", "normal"]
    session: str  # E.g. "s001"

    # Class variables
    description: tp.ClassVar[str] = (
        "Subset of TUH-EEG with labels for normal / abnormal recordings"
    )
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Lopez2017,
        title={Automated Identification of Abnormal Adult EEGs},
        author={López S},
        year={2017},
        publisher={Temple University}
    }
        """
    doi: tp.ClassVar[str] = "doi:10.1109/SPMB.2015.7405423"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""

        folder = Path(path) / "edf"
        assert folder.exists()

        for split_dir in folder.iterdir():
            split = split_dir.name
            for label_dir in split_dir.iterdir():
                label = label_dir.name
                for file_path in label_dir.rglob("*.edf"):
                    subject, session, token_number = file_path.stem.split("_")
                    yield cls(
                        path=str(path),
                        subject=subject,
                        split=split,  # type: ignore
                        label=label,  # type: ignore
                        session=session,
                        token_number=token_number,
                    )

    def _get_eeg_filename(self) -> str:
        eeg_file = str(
            Path(self.path)
            / "edf"
            / self.split
            / self.label
            / self.channel_configuration
            / f"{self.subject}_{self.session}_{self.token_number}.edf"
        )
        return eeg_file


class Hamid2020(_BaseTuhEeg):
    token_number: str  # E.g. "t000"
    session: str  # E.g. "s001"
    channel_configuration: tp.Literal[
        "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
    ]

    # Class variables
    description: tp.ClassVar[str] = "Subset of TUH-EEG with artifact events"
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_artifact/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Hamid2020,
        author={Hamid, A. and Gagliano, K. and Rahman, S. and Tulin, N. and Tchiong, V. and Obeid, I. and Picone, J.},
        booktitle={2020 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        title={The Temple University Artifact Corpus: An Annotated Corpus of EEG Artifacts},
        year={2020},
        pages={1--4},
    }
    """
    doi: tp.ClassVar[str] = "10.1109/SPMB50085.2020.9353647"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        folder = Path(path) / "edf"
        for channel_folder in folder.iterdir():
            channel_configuration = channel_folder.stem
            for file_path in channel_folder.rglob("*.edf"):
                subject, session, token_number = file_path.stem.split("_")
                yield cls(
                    path=str(path),
                    channel_configuration=channel_configuration,  # type: ignore
                    subject=subject,
                    session=session,
                    token_number=token_number,
                )

    def _get_eeg_filename(self) -> str:
        file_dir = Path(self.path) / "edf" / self.channel_configuration
        eeg_file = file_dir / f"{self.subject}_{self.session}_{self.token_number}.edf"
        return str(eeg_file)

    def _load_events(self) -> pd.DataFrame:
        file_dir = Path(self.path) / "edf" / self.channel_configuration
        events_file = str(
            file_dir / f"{self.subject}_{self.session}_{self.token_number}.csv"
        )
        seiz_events_file = str(
            file_dir / f"{self.subject}_{self.session}_{self.token_number}_seiz.csv"
        )

        events_df = self._format_artifact_events(events_file, seiz_events_file)

        return pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "type": "Eeg",
                            "start": 0.0,
                            "filepath": f"method:_load_raw?timeline={self.timeline}",
                        }
                    ]
                ),
                events_df,
            ],
            ignore_index=True,
        )

    def _load_events_csv(
        self,
        file_path: str,
        header_row: int = 6,
    ) -> pd.DataFrame:
        events_df = pd.read_csv(file_path, header=header_row)
        return events_df

    def _format_artifact_events(
        self,
        file_path: str,
        seiz_path: str,
    ) -> pd.DataFrame:
        out_cols = [
            "type",
            "start",
            "duration",
            "filepath",
            "channel",
            "state",
            "had_epileptic_event",
        ]
        events_df = self._load_events_csv(file_path)
        events_df["type"] = "Artifact"
        if Path(seiz_path).exists():
            seiz_df = self._load_events_csv(seiz_path)
            seiz_df["type"] = "Seizure"
            events_df = pd.concat([events_df, seiz_df])
            had_epileptic_event = True
        else:
            had_epileptic_event = False
        output = events_df[["channel", "start_time", "label", "type"]]
        output = output.rename(columns={"start_time": "start", "label": "state"})
        output["filepath"] = ""  # no stimulus
        output["had_epileptic_event"] = had_epileptic_event
        # Compute duration
        output["duration"] = events_df["stop_time"] - events_df["start_time"]
        # Separate combo events (e.g., 'eyem_musc') into multiple rows
        output["state"] = output.state.str.split("_")
        output = output.explode(column="state")
        output["state"] = output["state"].replace({"elec": "elpp"})
        return output[out_cols]


class Veloso2017(_BaseTuhEeg):
    token_number: str  # E.g. "t000"
    label: tp.Literal["epilepsy", "no_epilepsy"]
    session: str  # E.g. "s001"
    date: str  # YYYY or YYYY_MM_DD, E.g., 2000
    channel_configuration: tp.Literal[
        "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
    ]

    # Class variables
    description: tp.ClassVar[str] = (
        "Subset of TUH-EEG with labels no epilepsy / epilepsy recordings"
    )
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_epilepsy/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Veloso2017,
        author={Veloso, L. and McHugh, J. and von Weltin, E. and Lopez, S. and Obeid, I. and Picone, J},
        title={Big data resources for EEGs: Enabling deep learning research},
        booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        year={2017},
        pages={1--3},
    }
    """
    doi: tp.ClassVar[str] = "10.1109/SPMB.2017.8257044"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        folder = Path(path)
        for label_dir in folder.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name[3:]
            for sub_dir in label_dir.iterdir():
                for sess_dir in sub_dir.iterdir():
                    date = sess_dir.stem[5:]
                    for file_path in sess_dir.rglob("*.edf"):
                        channel_configuration = file_path.parts[-2]
                        subject, session, token_number = file_path.stem.split("_")
                        yield cls(
                            path=str(path),
                            subject=subject,
                            label=label,  # type: ignore
                            session=session,
                            date=date,
                            channel_configuration=channel_configuration,  # type: ignore
                            token_number=token_number,
                        )

    def _get_eeg_filename(self) -> str:
        ep_label_dict = {"epilepsy": "00_epilepsy", "no_epilepsy": "01_no_epilepsy"}
        eeg_file = str(
            Path(self.path)
            / f"{ep_label_dict[self.label]}"
            / self.subject
            / f"{self.session}_{self.date}"
            / self.channel_configuration
            / f"{self.subject}_{self.session}_{self.token_number}.edf"
        )
        return eeg_file


class Harati2015(_BaseTuhEeg):
    token_number: str | None  # E.g. "t000"
    split: tp.Literal["train", "eval"]
    label: (
        tp.Literal[
            "bckg",  # Background (no seizure)
            "gped",  # Generalized periodic epileptiform discharges
            "pled",  # Periodic lateralized epileptiform discharges
            "spsw",  # Spike and/or sharp waves
        ]
        | None
    )
    session: str | None

    # Class variables
    description: tp.ClassVar[str] = (
        "Subset of TUH-EEG with annotations for epilepsy and artifact events"
    )
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Harati2015,
        title={Improved EEG event classification using differential energy},
        author={Harati, Amir and Golmohammadi, Meysam and Lopez, Silvia and Obeid, Iyad and Picone, Joseph},
        booktitle={2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        pages={1--4},
        year={2015},
        organization={IEEE}
    }
    """
    doi: tp.ClassVar[str] = "10.1109/SPMB.2015.7405421"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        folder = Path(path) / "edf"
        for split_dir in folder.iterdir():
            split = split_dir.name
            for sub_dir in split_dir.iterdir():
                for file_path in sub_dir.rglob("*.edf"):
                    if split == "train":
                        subject, token_number = file_path.stem.split("_")
                        label = None
                        session = None
                    elif split == "eval":
                        label, subject, _, _ = file_path.stem.split("_")
                        session = file_path.stem[9:]
                        token_number = None
                    yield cls(
                        path=str(path),
                        subject=subject,
                        split=split,  # type: ignore
                        token_number=token_number,  # type: ignore
                        label=label,  # type: ignore
                        session=session,  # type: ignore
                    )

    def _get_eeg_filename(self) -> str:
        eeg_dir = Path(self.path) / "edf" / self.split / self.subject
        if self.split == "eval":
            eeg_file = str(eeg_dir / f"{self.label}_{self.subject}_{self.session}.edf")
        else:
            eeg_file = str(eeg_dir / f"{self.subject}_{self.token_number}.edf")
        return eeg_file

    def _get_rec_filename(self) -> str:
        subj_dir = Path(self.path) / "edf" / self.split / self.subject
        if self.split == "eval":
            rec_file = str(subj_dir / f"{self.label}_{self.subject}_{self.session}.rec")
        else:
            rec_file = str(subj_dir / f"{self.subject}_{self.token_number}.rec")
        return rec_file

    def _load_events(self) -> pd.DataFrame:

        events_df = self._load_annot_events()

        return pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "type": "Eeg",
                            "start": 0.0,
                            "filepath": f"method:_load_raw?timeline={self.timeline}",
                        }
                    ]
                ),
                events_df,
            ],
            ignore_index=True,
        )

    def _load_annot_events(self) -> pd.DataFrame:
        rec_file = self._get_rec_filename()
        event_df = pd.DataFrame(
            np.genfromtxt(rec_file, delimiter=","),
            columns=["channel", "start", "stop", "label_code"],
        )
        label_dict = {
            1: "spsw",
            2: "gped",
            3: "pled",
            4: "eyem",
            5: "artf",
            6: "bckg",
        }
        event_df["state"] = event_df.label_code.map(label_dict)
        event_df.start = event_df.start.round(1)
        event_df.stop = event_df.stop.round(1)
        return self._condense_events(event_df[["start", "stop", "channel", "state"]])

    def _condense_events(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """Annotations are labeled for every sec; group consecutive events and set duration."""
        output = pd.DataFrame(
            columns=["type", "start", "duration", "channel", "state", "filepath"]
        )
        event_times = pd.concat(
            [
                event_df.groupby(by=["channel", "state"])["start"].unique(),
                event_df.groupby(by=["channel", "state"])["stop"].unique(),
            ],
            axis=1,
        )

        channel: float
        state: str
        for (channel, state), data in event_times.iterrows():  # type: ignore
            # group start events within 1s
            start_times = set(data.start) - set(data.stop)
            for t in sorted(data.start):
                start_times.difference_update(
                    set(np.linspace(t + 0.1, t + 1, 10).round(1))
                )
            # group stops events within 1s
            stop_times = set(data.stop) - set(data.start)
            for t in sorted(data.stop, reverse=True):
                stop_times.difference_update(
                    set(np.linspace(t - 1, t - 0.1, 10).round(1))
                )
            if state in ["eyem", "artf"]:
                event_class = "Artifact"
            else:
                event_class = "EpileptiformActivity"
            for start_t, stop_t in zip(sorted(start_times), sorted(stop_times)):
                duration = stop_t - start_t
                assert duration > 0
                output.loc[len(output)] = [
                    event_class,
                    start_t,
                    duration,
                    channel,
                    state,
                    "",
                ]
        return output


class VonWeltin2017(_BaseTuhEeg):
    token_number: str  # E.g. "t000"
    session: str  # E.g. "s001"
    date: str  # YYYY or YYYY_MM_DD, E.g., 2000
    channel_configuration: tp.Literal[
        "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
    ]

    # Class variables
    description: tp.ClassVar[str] = (
        "Subset of TUH-EEG with annotations for slowing events"
    )
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_slowing/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{VonWeltin2017,
        author={Von Weltin, E. and Ahsan, T. and Shah, V. and Jamshed, D. and Golmohammadi, M. and Obeid, I. and Picone, J.},
        booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        title={Electroencephalographic slowing: A primary source of error in automatic seizure detection},
        year={2017},
        pages={1-5},
    }
    """
    doi: tp.ClassVar[str] = "10.1109/SPMB.2017.8257018"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        folder = Path(path) / "edf"
        for sub_dir in folder.iterdir():
            for sess_dir in sub_dir.iterdir():
                date = sess_dir.stem[5:]
                for file_path in sess_dir.rglob("*.edf"):
                    channel_configuration = file_path.parts[-2]
                    subject, session, token_number = file_path.stem.split("_")
                    yield cls(
                        path=str(path),
                        subject=subject,
                        session=session,
                        date=date,
                        channel_configuration=channel_configuration,  # type: ignore
                        token_number=token_number,
                    )

    def _get_eeg_filename(self) -> str:
        eeg_file = str(
            Path(self.path)
            / "edf"
            / self.subject
            / f"{self.session}_{self.date}"
            / self.channel_configuration
            / f"{self.subject}_{self.session}_{self.token_number}.edf"
        )
        return eeg_file


class Shah2018(_BaseTuhEeg):
    split: tp.Literal["dev", "train", "eval"]
    token_number: str  # E.g. "t000"
    session: str  # E.g. "s001"
    date: str  # YYYY or YYYY_MM_DD, E.g., 2000

    channel_configuration: tp.Literal[
        "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
    ]

    # Class variables
    description: tp.ClassVar[str] = (
        "Subset of TUH-EEG with extensive annotations for seizure and artifact events"
    )
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_seizure/"
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Shah2018,
        author={Shah, Vinit  and von Weltin, Eva  and Lopez, Silvia  and McHugh, James Riley  and Veloso, Lillian  and Golmohammadi, Meysam  and Obeid, Iyad  and Picone, Joseph },
        title={The Temple University Hospital Seizure Detection Corpus},
        journal={Frontiers in Neuroinformatics},
        volume={12},
        year={2018},
    }
    """
    doi: tp.ClassVar[str] = "10.3389/fninf.2018.00083"

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""
        folder = Path(path) / "edf"
        for split_dir in folder.iterdir():
            split = split_dir.name
            for sub_dir in split_dir.iterdir():
                for sess_dir in sub_dir.iterdir():
                    date = sess_dir.stem[5:]
                    for file_path in sess_dir.rglob("*.edf"):
                        channel_configuration = file_path.parts[-2]
                        subject, session, token_number = file_path.stem.split("_")
                        yield cls(
                            path=str(path),
                            subject=subject,
                            session=session,
                            date=date,
                            split=split,  # type: ignore
                            token_number=token_number,
                            channel_configuration=channel_configuration,  # type: ignore
                        )

    def _get_eeg_filename(self) -> str:
        eeg_file = str(
            Path(self.path)
            / "edf"
            / self.split
            / self.subject
            / f"{self.session}_{self.date}"
            / self.channel_configuration
            / f"{self.subject}_{self.session}_{self.token_number}.edf"
        )
        return eeg_file

    # TODO: create custom _load_events to add annotation events
