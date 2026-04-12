# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pandas as pd

from ..data import BaseData
from ..download import Datalad, Wildcard


class Courtois2020(BaseData):
    # timeline
    session: str
    # contains task inside the move (e.g. episode / season)
    task: str
    # stores the movie to be watched
    movie: str = "friends"

    # study/class level
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://www.biorxiv.org/content/10.1101/2023.09.06.556533v1"
    licence: tp.ClassVar[str] = "CC0"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()
    bibtex: tp.ClassVar[
        str
    ] = """
    @article {courtois2020,
        author = {Maelle Freteault and Loic Tetrel and Maximilien Le Clei and Pierre Bellec and Nicolas Farrugia},
        title = {Aligning the activity of artificial and biological neural networks to build personalised models of auditory processing in a massive individual fMRI dataset},
        elocation-id = {2023.09.06.556533},
        year = {2023},
        doi = {10.1101/2023.09.06.556533},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2023/09/06/2023.09.06.556533},
        eprint = {https://www.biorxiv.org/content/early/2023/09/06/2023.09.06.556533.full.pdf},
        journal = {bioRxiv}
    }
    """

    @classmethod
    def _download(cls, path: Path, movie: str = "friends") -> None:
        """Download all subject in the dataset"""
        Datalad(
            study="courtois",
            dset_dir=path,
            repo_url=f"https://github.com/courtois-neuromod/{movie}.fmriprep.git",
            threads=20,
            folders=[Wildcard(folder="sub-*")],
        ).download()
        Datalad(
            study="courtois",
            dset_dir=path,
            repo_url=f"https://github.com/courtois-neuromod/{movie}.stimuli.git",
            threads=20,
            folders=[Wildcard(folder="s*")],
        ).download()

    @classmethod
    def _iter_timelines(
        cls, path: Path | str, movie: str = "friends", skip_missing: bool = True
    ):
        path = Path(path)
        # loop across subjects
        for subject_path in (path / f"{movie}.fmriprep").glob("sub-*"):
            sub = subject_path.name
            if not subject_path.is_dir():
                continue

            # loop across sessions
            for session_path in (subject_path).glob("ses-*"):
                ses = session_path.name
                if not session_path.is_dir():
                    continue
                # loop across recordings
                for bold_path in (session_path / "func").glob(
                    "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                ):
                    # check stim file
                    task_name = bold_path.name.split("_")[2][5:]
                    season = int(task_name[1:3])
                    stim_path = (
                        path
                        / f"{movie}.stimuli"
                        / f"s{season}"
                        / f"{movie}_{task_name}.mkv"
                    )

                    # check for skipping
                    # NOTE: relevant since not all courtois data is public
                    if not bold_path.exists() and skip_missing:
                        continue

                    # sanity checks
                    assert (
                        bold_path.exists() and bold_path.is_file()
                    ), f"Bold data ({bold_path}) does not exist"
                    assert (
                        stim_path.exists() and stim_path.is_file()
                    ), f"Stimuli ({stim_path}) does not exist"

                    yield cls(subject=sub, session=ses, task=task_name, path=path, movie=movie)  # type: ignore

    def _file(self, suffix: str) -> Path:
        """Get file from its suffix"""
        folder = (
            Path(self.path)
            / f"{self.movie}.fmriprep"
            / self.subject
            / self.session
            / "func"
        )
        fname = "_".join(
            [
                self.subject,
                self.session,
                f"task-{self.task}",
                "space-MNI152NLin2009cAsym",
                "desc-preproc",
                suffix,
            ]
        )
        return folder / fname

    def _load_events(self) -> pd.DataFrame:
        """Load events"""
        # generate stim path
        season = int(self.task[1:3])
        stim_path = (
            Path(self.path)
            / f"{self.movie}.stimuli"
            / f"s{season}"
            / f"{self.movie}_{self.task}.mkv"
        )

        # generate dataframe
        # FIXME: add frequencies here?
        evs = [
            dict(type="Fmri", start=0, filepath=self._file("bold.nii.gz")),
            dict(type="Video", start=0, filepath=stim_path),
        ]
        return pd.DataFrame.from_records(evs)
