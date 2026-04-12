# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import nibabel
import pandas as pd

from neuralset import BaseData
from neuralset.download import Datalad, Wildcard
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


class Hebart2023Bold(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004192/versions/1.0.5"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{hebart2023things,
        title={THINGS-data, a multimodal collection of large-scale datasets for
        investigating object representations in human brain and behavior},
        author={Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter,
        Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna
        and Vaziri-Pashkam, Maryam and Baker, Chris I},
        journal={Elife},
        volume={12},
        pages={e82580},
        year={2023},
        publisher={eLife Sciences Publications Limited}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.18112/openneuro.ds004192.v1.0.5"
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "BOLD data for 3 subjects watching still images in 3T fMRI"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("datalad>=0.19.5",)

    SUBJECTS: tp.ClassVar[tp.Tuple[int, ...]] = (1, 2, 3)

    SESSIONS_PER_SUBJECT: tp.ClassVar[int] = 12
    RUNS_PER_SESSION: tp.ClassVar[int] = 10

    BIDS_FOLDER: tp.ClassVar[str] = "download/ds004192"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives"

    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"

    TASK: tp.ClassVar[str] = "things"

    SESSION_SUFFIX: tp.ClassVar[str] = "things"

    TR_FMRI_S: tp.ClassVar[float] = 1.5

    session: int
    run: int

    @classmethod
    def _download(cls, path: Path) -> None:
        Datalad(
            study="hebart2023bold",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds004192.git",
            threads=4,
            folders=[
                Wildcard(folder="sub-*"),
            ],
        ).download()
        cls._write_test_categories(path / "hebart2023bold")

    @classmethod
    def _write_test_categories(cls, path: Path) -> None:
        event_dfs = []

        for subject, session, run in cls._iter_subject_session_run():
            bids_events_df_fp = get_bids_filepath(
                root_path=Path(path) / cls.BIDS_FOLDER,
                subject=subject,
                session=session,
                run=run,
                task=cls.TASK,
                filetype="events",
                data_type="Fmri",
                ses_suffix=cls.SESSION_SUFFIX,
            )
            event_dfs.append(read_bids_events(bids_events_df_fp))
        event_df = pd.concat(event_dfs, axis=0)
        test_trials = event_df[event_df.trial_type == "test"]
        test_categories = test_trials.file_path.map(cls._get_category).unique()
        with open(path / "test_categories.txt", mode="w", encoding="utf8") as f:
            f.writelines([f"{category}\n" for category in test_categories])

    @classmethod
    def _get_test_categories(cls, path: str | Path) -> tp.Set[str]:
        path = Path(path)
        with open(path / "test_categories.txt", mode="r", encoding="utf8") as f:
            return set([line.strip() for line in f])

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)

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
            run=self.run,
            task=self.TASK,
            filetype="events",
            data_type="Fmri",
            ses_suffix=self.SESSION_SUFFIX,
        )

        bids_events_df = read_bids_events(bids_events_df_fp)
        path_to_stimuli = (Path(self.path) / ".." / "THINGS-images").resolve(strict=False)
        test_cats = self._get_test_categories(self.path)
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df,
            path_to_stimuli,
            self._get_fmri_frequency(),
            test_cats,
        )

        # Add indicator column for events of test categories
        ns_events_df["stem"] = ns_events_df.filepath.apply(lambda x: Path(x).stem)
        ns_events_df["is_test_category"] = ns_events_df.category.isin(test_cats)
        # For consistency with other THINGS-derived studies, redefine categories and split
        ns_events_df.category = ns_events_df.stem.apply(
            lambda x: "_".join(x.split("_")[:-1])
        )
        ns_events_df["split"] = ns_events_df.hebart2023_paper_split

        # Add shared filepath
        shared_things_path = (Path(self.path) / ".." / "THINGS-images").resolve(
            strict=False
        )
        if shared_things_path.exists():
            ns_events_df["shared_filepath"] = (
                str(shared_things_path)
                + "/"
                + ns_events_df.category
                + "/"
                + ns_events_df.stem
                + ".jpg"
            )

        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: str) -> nibabel.Nifti1Image:
        return get_masked_bold_image(self._get_bold_image(), self._get_bold_mask())

    @classmethod
    def _iter_subject_session_run(cls):
        for subject in cls.SUBJECTS:
            for session in range(1, cls.SESSIONS_PER_SUBJECT + 1):
                for run in range(1, cls.RUNS_PER_SESSION + 1):
                    yield (subject, session, run)

    @classmethod
    def _get_category(cls, file_path: str) -> str:
        return re.sub(r"\d", "", " ".join(Path(file_path).stem.split("_")[:-1]))

    def _get_ns_img_events_df(
        cls,
        bids_events_df: pd.DataFrame,
        stimuli_path: str | Path,
        frequency: float,
        test_cats: tp.Set[str],
    ) -> pd.DataFrame:
        # Leave out 'catch' trials (used for making sure subject is focused)
        bids_events_df = bids_events_df[bids_events_df.trial_type != "catch"]
        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            parent = "_".join(Path(bids_event["file_path"]).stem.split("_")[:-1])
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                frequency=frequency,
                filepath=str(
                    Path(stimuli_path) / parent / Path(bids_event["file_path"]).name
                ),
                hebart2023_paper_split=(
                    "test" if bids_event["trial_type"] == "test" else "train"
                ),
                category=cls._get_category(bids_event["file_path"]),
            )
            ns_events.append(ns_event)

        ns_events_df = pd.DataFrame(ns_events)

        # Add zero-shot split, that is:
        # Remove images from train-set whose category is in test
        ns_events_df["zero_shot_split"] = ns_events_df.hebart2023_paper_split
        ns_events_df.loc[
            (ns_events_df.hebart2023_paper_split == "train")
            & (ns_events_df.category.isin(test_cats)),
            "zero_shot_split",
        ] = "trash"

        # Add large split, that is:
        # from zero-shot split, add 'trash' images to 'test'
        ns_events_df["large_split"] = ns_events_df.zero_shot_split
        ns_events_df.loc[
            ns_events_df.zero_shot_split == "trash",
            "large_split",
        ] = "test"
        return ns_events_df

    def _get_bold_mask(self):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            subject=self.subject,
            session=self.session,
            run=self.run,
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
