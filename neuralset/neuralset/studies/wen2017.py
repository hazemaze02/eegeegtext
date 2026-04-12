# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pandas as pd

from ..data import BaseData


def _get_nii_file(path: Path | str, subject: str, seg: str, fmri_run: int) -> Path:
    path = Path(path)
    seg_dir = path / subject / "fmri" / seg
    nii = seg_dir / "mni" / f"{seg}_{fmri_run}_mni.nii.gz"
    # Outrageously, some test files have a different
    # naming convention...
    if not nii.exists():
        nii = seg_dir / "mni" / f"{seg}_{fmri_run}.mni.nii.gz"
    assert nii.exists(), f"Missing file {nii} for {subject!r} and {seg!r}"
    return nii


def _get_video_file(path: Path | str, seg: str) -> Path:
    path = Path(path)
    return path / f"stimuli/{seg}.mp4"


class Wen2017(BaseData):
    seg: str
    run: int

    # study/class level
    device: tp.ClassVar[str] = "Fmri"
    licence: tp.ClassVar[str] = "CC-BY 0"
    url: tp.ClassVar[str] = "https://academic.oup.com/cercor/article/28/12/4136/4560155"
    TR_FMRI_S: tp.ClassVar[float] = 2.0  # don't rely on nifti header

    # TODO: Add download method, get brainmetric downloader
    @classmethod
    def _download(cls, path: Path) -> None:
        # url = "https://purr.purdue.edu/publications/2809/1"
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        path = Path(path) / "download" / "video_fmri_dataset"
        # loop across subjects
        for subject_dir in path.iterdir():
            subject = subject_dir.name
            if not subject.startswith("subject") or not subject_dir.is_dir():
                continue

            # loop across recordings
            for seg_dir in (subject_dir / "fmri").iterdir():
                seg = seg_dir.name
                is_train = seg.startswith("seg")
                is_test = seg.startswith("test")
                # FIXME what is this?
                if not (is_train or is_test):
                    continue
                # check if video exists
                file = _get_video_file(path, seg)
                assert file.exists()

                # define fmri runs
                fmri_runs = range(1, 3) if is_train else range(1, 11)
                for run_ in fmri_runs:
                    # check file exist
                    nii = _get_nii_file(path, subject, seg, run_)
                    assert nii.exists()

                    yield cls(subject=subject, seg=seg, run=run_, path=path)  # type: ignore

    def _load_events(self) -> pd.DataFrame:
        import nibabel

        video_file = _get_video_file(self.path, self.seg)
        nii_file = _get_nii_file(self.path, self.subject, self.seg, self.run)
        nii: tp.Any = nibabel.load(nii_file, mmap=True)
        freq = 1.0 / self.TR_FMRI_S
        dur = nii.shape[-1] / freq
        return pd.DataFrame(
            [
                dict(type="Video", start=0, filepath=video_file),
                dict(
                    type="Fmri", start=0, filepath=nii_file, frequency=freq, duration=dur
                ),
            ]
        )
