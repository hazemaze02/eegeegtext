# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Adds the NMT EEG Scalp Dataset (Khan et al., 2022) annotated for pathological and normal.
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.755817/full

- Loads EEG data for N=2417 subjects
- Does not load N=1090 corrupted EEG files ("._*.edf")
- Doesn't exactly match the reported properties of the dataset.
  - Matches the reported subject count (N=2417)
  - Matches the reported sampling frequency (200 Hz) and EEG channels (19 + 2 reference)
  - Does not match the reported durations of the EEG recordings.(The dataset contains EEG recordings of variable duration.)
    - We find a mean duration of 12.67 min, whereas they report a mean duration of 15 min.
    - The histograms are similar but somewhat different.
See PR #803 for details (https://github.com/fairinternal/brainai/pull/803)
"""

import typing as tp
from pathlib import Path

import pandas as pd
from mne.datasets import fetch_dataset

from neuralset.data import BaseData


class Khan2022(BaseData):
    label: tp.Literal["abnormal", "normal"]
    split: tp.Literal["train", "eval"]
    subject: str

    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://pubmed.ncbi.nlm.nih.gov/35069095/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{khan2022,
        title={The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling},
        author={Khan, Hassan A and Ain, Rahat U and Kamboh, Awais M and Butt, Hammad T and Shafait, Saima and Alamgir, Wasim and Stricker, Didier and Shafait, Faisal},
        year={2022},
        journal={Front Neurosci},
        volume={15},
        pages={755817},
        publisher={Frontiers Media}
    }
        """
    doi: tp.ClassVar[str] = "doi: 10.3389/fnins.2021.755817"
    description: tp.ClassVar[str] = (
        "A corpus of 2,417 EEG recordings annotated as normal or abnormal."
    )

    @classmethod
    def _download(cls, path: Path) -> None:
        """
        Leverages mne.datasets.fetch_dataset method to download.
        - The reported download link is corrupted: https://dll.seecs.nust.edu.pk/downloads/
        - See approach in braindecode: https://github.com/braindecode/braindecode//blob/master/braindecode/datasets/nmt.py#L46-L172
        """
        NMT_URL = "https://zenodo.org/record/10909103/files/NMT.zip"
        NMT_archive_name = "NMT.zip"
        NMT_folder_name = "MNE-NMT-eeg-dataset"
        NMT_dataset_name = "NMT-EEG-Corpus"

        NMT_dataset_params = {
            "dataset_name": NMT_dataset_name,
            "url": NMT_URL,
            "archive_name": NMT_archive_name,
            "folder_name": NMT_folder_name,
            "hash": "77b3ce12bcaf6c6cce4e6690ea89cb22bed55af10c525077b430f6e1d2e3c6bf",
            "config_key": NMT_dataset_name,
        }
        path = fetch_dataset(
            dataset_params=NMT_dataset_params,
            path=path,
            processor="unzip",
            force_update=False,
        )

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        folder = Path(path) / "nmt_scalp_eeg_dataset"
        for label in ["normal", "abnormal"]:
            for split in ["train", "eval"]:
                split_dir = folder / label / split
                for file_path in split_dir.rglob("*.edf"):
                    subject = file_path.stem
                    # Skip bad files
                    if "._" in subject:
                        continue
                    yield cls(
                        subject=subject,
                        path=path,
                        label=label,  # type: ignore
                        split=split,  # type: ignore
                    )

    def _load_events(self) -> pd.DataFrame:
        eeg_file = (
            Path(self.path)
            / "nmt_scalp_eeg_dataset"
            / self.label
            / self.split
            / f"{self.subject}.edf"
        )
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=eeg_file,
                    split=self.split,
                    label=self.label,
                ),
            ]
        )
