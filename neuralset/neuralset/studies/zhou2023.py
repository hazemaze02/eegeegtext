# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import pandas as pd

from ..data import BaseData
from ..download import Datalad, Wildcard


class Zhou2023(BaseData):
    # timeline
    session: str
    fmri_run: str
    TR_FMRI_S: tp.ClassVar[float] = 2.0  # don't rely on nifti header

    # study/class level
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://www.nature.com/articles/s41597-023-02325-6"
    licence: tp.ClassVar[str] = "CC-BY 0"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{zhou2023large,
        title={A large-scale fMRI dataset for human action recognition},
        author={Zhou, Ming and Gong, Zhengxin and Dai, Yuxuan and Wen, Yushan
                and Liu, Youyi and Zhen, Zonglei},
        journal={Scientific Data},
        volume={10},
        number={1},
        pages={415},
        year={2023},
        publisher={Nature Publishing Group UK London}
    }
    """

    @classmethod
    def _download(cls, path: Path) -> None:
        """Download all subjects in the dataset"""
        Datalad(
            study="zhou",
            dset_dir=path,
            repo_url="https://github.com/OpenNeuroDatasets/ds004488.git",
            threads=4,
            folders=[Wildcard(folder="sub-*"), "stimuli"],
        ).download()

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        path = Path(path)
        # loop across subjects
        folder = path / "ds004488"
        if not folder.exists():
            raise ValueError(f"No folder {folder}")
        for subject_path in folder.glob("sub-*"):
            if not subject_path.is_dir():
                continue

            # loop across recordings
            sub_data_path = subject_path / "ses-action01" / "func"
            for bold in sub_data_path.glob("*_bold.nii.gz"):
                # retrieve names
                filtr = r"sub-(\d+)_ses-action(\d+)_"
                filtr += r"task-action_run-(\d+)_bold.nii.gz"
                match = re.match(filtr, bold.name)
                if match is None:
                    raise RuntimeError(f"Could not parse {bold.name}")
                subject, session, fmri_run = match.groups()
                subject = "sub-" + subject
                session = "ses-action" + session
                fmri_run = "run-" + fmri_run
                assert subject == subject_path.name

                yield cls(subject=subject, session=session, fmri_run=fmri_run, path=path)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        assert self._file("events.tsv").exists()
        assert self._file("bold.json").exists()

    def _file(self, suffix: str) -> Path:
        """Get file from its suffix"""
        folder = Path(self.path) / "ds004488" / self.subject / "ses-action01" / "func"
        fname = "_".join(
            [self.subject, self.session, "task-action", self.fmri_run, suffix]
        )
        return folder / fname

    def _load_events(self) -> pd.DataFrame:
        """Load events"""
        import nibabel

        # video events
        events = pd.read_csv(self._file("events.tsv"), sep="\t")
        stim_path = Path(self.path) / "ds004488" / "stimuli"
        events["type"] = "Video"
        events = events.rename(columns={"onset": "start", "stim_file": "filepath"})
        events["filepath"] = stim_path / events["filepath"]

        # add fmri event
        fp = self._file("bold.nii.gz")
        if not fp.exists():
            raise ValueError(f"Missing bold file {fp}")

        nii: tp.Any = nibabel.load(fp, mmap=True)
        freq = 1.0 / self.TR_FMRI_S
        dur = nii.shape[-1] / freq
        fmri = dict(type="Fmri", start=0, filepath=fp, frequency=freq, duration=dur)
        out = pd.concat([events, pd.DataFrame([fmri])], ignore_index=True)
        return out.reset_index(drop=True)
