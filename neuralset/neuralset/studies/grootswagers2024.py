# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralset.data import BaseData
from neuralset.download import Datalad, Wildcard

logger = logging.getLogger(__name__)
logger.propagate = False


class Grootswagers2024(BaseData):
    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004357/versions/1.0.1"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{grootswagers2024,
        title={Mapping the dynamics of visual feature coding: Insights into perception and integration},
        author={Grootswagers, Tijl and Robinson, Amanda K and Shatek, Sophia M and Carlson, Thomas A},
        journal={PLoS Comput Biol},
        volume={20},
        number={1},
        pages={e1011760},
        year={2024},
        publisher={PLOS}
    }
    """
    doi: tp.ClassVar[str] = "https://doi.org/10.1371/journal.pcbi.1011760"
    licence: tp.ClassVar[str] = ""
    description: tp.ClassVar[str] = "16 subjects watching still gabor-like images in EEG."
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
    )

    TASK: tp.ClassVar[str] = "task-rsvp"

    @classmethod
    def _download(cls, path: Path) -> None:
        # Download MEG data from OpenNeuro
        Datalad(
            study="grootswagers2024",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds004357.git",
            threads=4,
            folders=[
                Wildcard(folder="sub-*/eeg"),
            ],
        ).download()
        # Download stimuli
        Datalad(
            study="grootswagers2024",
            dset_dir=f"{path}",
            repo_url="https://github.com/Tijl/features-eeg",
            threads=4,
            folders=[
                Wildcard(folder="stimuli/stim*.png"),
            ],
        ).download()

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        for subject in [f"sub-{i:02}" for i in range(1, 17)]:
            yield cls(subject=subject, path=path)

    def _get_filenames(self) -> tuple[Path, Path]:
        folder = Path(self.path) / "download" / "ds004357" / self.subject / "eeg"
        vhdr_file = folder / f"{self.subject}_{self.TASK}_eeg.vhdr"
        events_file = folder / f"{self.subject}_{self.TASK}_events.tsv"
        return vhdr_file, events_file

    def _load_events(
        self,
    ) -> pd.DataFrame:
        eeg_events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    filepath=f"method:_load_raw?timeline={self.timeline}",
                    start=0.0,
                ),
            ]
        )
        events_file = self._get_filenames()[1]
        events_df = self._format_image_events(events_file)
        stimulus_df = self._format_stimulus_events(events_file)
        return pd.concat([eeg_events, events_df, stimulus_df]).reset_index()

    def _format_image_events(
        self,
        file_path: str | Path,
    ) -> pd.DataFrame:
        raw_events_df = pd.read_csv(file_path, sep="\t")
        events_df = pd.DataFrame(index=raw_events_df.index)
        events_df["filepath"] = raw_events_df["stimname"].apply(
            lambda x: f"{self.path}/download/features-eeg/stimuli/{x}"
        )
        events_df["type"] = "Image"
        events_df["start"] = raw_events_df["onset"]
        events_df["duration"] = (
            raw_events_df["time_stimoff"] - raw_events_df["time_stimon"]
        )
        events_df["orientation"] = raw_events_df["feature_ori"]
        events_df["spatial_frequency"] = raw_events_df["feature_sf"]
        events_df["rgb_color"] = raw_events_df["feature_color"]
        events_df["contrast"] = raw_events_df["feature_contrast"]
        return events_df

    def _format_stimulus_events(
        self,
        file_path: str | Path,
    ) -> pd.DataFrame:
        events_file = self._get_filenames()[1]
        events_df = self._format_image_events(events_file)
        feat_mapping = {
            "orientation": {
                22.5: 0,
                67.5: 1,
                112.5: 2,
                157.5: 3,
            },
            "spatial_frequency": {
                0.010: 0,
                0.025: 1,
                0.040: 2,
                0.055: 3,
            },
            "rgb_color": {
                "(66, 10, 104)": "0",
                "(147, 38, 103)": "1",
                "(221, 81, 58)": "2",
                "(252, 165, 10)": "3",
            },
            "contrast": {
                0.3: 0,
                0.5: 1,
                0.7: 2,
                0.9: 3,
            },
        }
        feat_dfs = []
        for feat in ["orientation", "spatial_frequency", "rgb_color", "contrast"]:
            feat_df = pd.DataFrame(
                columns=["type", "code", "description", "start", "duration"]
            )
            feat_df["code"] = events_df[feat].replace(feat_mapping[feat]).astype(float)  # type: ignore
            feat_df["start"] = events_df["start"]
            feat_df["duration"] = events_df["duration"]
            feat_df["type"] = "Stimulus"
            feat_df["description"] = feat
            feat_dfs.append(feat_df)
        return pd.concat(feat_dfs).reset_index(drop=True)

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        vhdr_file = self._get_filenames()[0]
        raw = mne.io.read_raw_brainvision(vhdr_file)
        montage = mne.channels.make_standard_montage("standard_1005")
        to_drop = [ch for ch in raw.ch_names if ch not in montage.ch_names]
        raw = raw.drop_channels(to_drop)
        if len(to_drop) > 0:
            logger.info("Dropped %s unrecognized EEG channels: %s", len(to_drop), to_drop)
        raw = raw.set_montage(montage, on_missing="ignore")
        return raw
