# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import mne
import pandas as pd

from ..data import BaseData


# from https://github.com/braindecode/braindecode/blob/0d4f70f42e05d94e1ac038b3c8478037ccd00938/braindecode/datasets/moabb.py#L19
def _find_dataset_in_moabb(
    dataset_name: str, dataset_kwargs: dict[str, tp.Any] | None = None
) -> tp.Any:
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list

    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            if dataset_kwargs is None:
                return dataset()
            else:
                return dataset(**dataset_kwargs)
    raise ValueError(f"{dataset_name} not found in moabb datasets")


class MOABBDataset2024(BaseData):
    subject: str
    session: str
    run: str

    # Study level
    device: tp.ClassVar[str] = "Eeg"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("moabb",)
    dataset_name: tp.ClassVar[str]

    @classmethod
    def _download(cls, path: Path):
        from moabb.datasets.base import CacheConfig

        # Download data and store all infornmation about
        # subjects, sessions, runs in one csv
        # Store all events in another csv
        set_dir = path / cls.dataset_name
        all_events_filepath = set_dir / "all_events.csv"
        if Path(all_events_filepath).exists():
            return
        dataset = _find_dataset_in_moabb(cls.dataset_name)

        # Load data
        runs_dict = dataset.get_data(
            subjects=None, cache_config=CacheConfig(path=set_dir)
        )

        # Store subjects sessions runs csv
        parts = []
        for subject in runs_dict:
            for session in runs_dict[subject]:
                for run in runs_dict[subject][session]:
                    parts.append(
                        dict(subject=str(subject), session=str(session), run=str(run))
                    )
        subject_session_run_df = pd.DataFrame(parts)

        set_dir.mkdir(parents=True, exist_ok=True)
        subject_session_run_df.to_csv(set_dir / "subject_session_run.csv")

        # Store events csv
        dfs = []

        previous_mapping = None
        for subject in runs_dict:
            for session in runs_dict[subject]:
                for run in runs_dict[subject][session]:
                    raw = runs_dict[subject][session][run]
                    # Could this lead to inconsistent mappings across subjects?
                    _, mapping = mne.events_from_annotations(raw)
                    if previous_mapping is None:
                        previous_mapping = mapping
                    assert mapping == previous_mapping
                    df = (
                        pd.DataFrame(raw.annotations)
                        .rename(columns=dict(onset="start"))
                        .drop("orig_time", axis=1)
                    )
                    #  is this always correct?
                    df["code"] = df.description.apply(lambda d: mapping[d]) - 1
                    df["type"] = "Stimulus"
                    df.modality = None

                    # Resave the data as fif
                    fif_dir = set_dir / f"{subject}_{session}_{run}"
                    fif_path = fif_dir / "raw.fif"
                    fif_dir.mkdir(parents=True, exist_ok=True)
                    # maybe remove overwrite later again for safety
                    raw.save(fif_path, overwrite=True)
                    uri = f"{fif_path}"
                    eeg = {"type": "Eeg", "filepath": uri, "start": 0}
                    df = pd.concat([pd.DataFrame([eeg]), df])
                    df["subject"] = str(subject)
                    df["session"] = str(session)
                    df["run"] = str(run)
                    dfs.append(df)
        overall_df = pd.concat(dfs)
        overall_df.to_csv(all_events_filepath)
        print(f"Saved to {all_events_filepath}")

    @classmethod
    def _iter_timelines(cls, path: Path | str) -> tp.Iterator["MOABBDataset2024"]:
        set_dir = path
        subject_session_run_df = pd.read_csv(
            Path(set_dir) / "subject_session_run.csv", index_col=0
        )
        for row in subject_session_run_df.itertuples():
            yield cls(
                subject=str(row.subject),
                session=str(row.session),
                run=str(row.run),
                path=path,
            )

    def _load_events(self) -> pd.DataFrame:
        # Subselect relevant dataframe from dataframe with all events
        set_dir = self.path
        overall_df = pd.read_csv(Path(set_dir) / "all_events.csv", index_col=0)
        part_df = overall_df[
            (overall_df.subject.astype(str) == self.subject)
            & (overall_df.session.astype(str) == self.session)
            & (overall_df.run.astype(str) == self.run)
        ].drop(["subject", "session", "run"], axis=1)
        return part_df


class Schirrmeister2017(MOABBDataset2024):
    dataset_name: tp.ClassVar[str] = "Schirrmeister2017"
    description: tp.ClassVar[
        str
    ] = """
    14 subjects performed 1040 trials of 4-second executed movements (left hand, right hand, feet or rest).
    """
