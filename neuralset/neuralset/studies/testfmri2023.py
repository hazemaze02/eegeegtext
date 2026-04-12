# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

from ..data import BaseData


class TestFmri2023(BaseData):
    # study/class level
    device: tp.ClassVar[str] = "Fmri"

    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["TestFmri2023"]:
        for i in range(3):
            yield cls(subject=str(i), path=path)

    def _load_raw(self, timeline: str) -> nibabel.filebasedimages.FileBasedImage:
        # pylint: disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        nii = Path(self.path) / f"sub-{self.subject}.nii.gz"

        if not nii.exists():
            n_voxels = 20
            n_times = 10
            # lets vary number of voxels across subjects
            n_voxels += int(self.subject)
            data_array = np.random.rand(n_voxels, n_voxels, n_voxels, n_times)
            nifti_image = nibabel.Nifti1Image(data_array, np.eye(4))
            tr_in_seconds = 2.0
            nifti_image.header["pixdim"][4] = tr_in_seconds
            nii.parent.mkdir(exist_ok=True)
            nibabel.save(nifti_image, nii)

        return nibabel.load(nii)

    def _load_events(self) -> pd.DataFrame:
        sentences = "hello world. the quick brown fox. they quit. good bye."

        events = []
        start = 1.0
        splits = ["train", "test", "val"]
        for sid, sentence in enumerate(sentences.split(".")):
            if not sentence:
                continue
            sentence += "."
            for word in sentence.split():
                events.append(
                    dict(
                        start=start,
                        text=word,
                        duration=len(word) / 30,
                        type="Word",
                        language="english",
                        modality="audio",
                        split=splits[sid % 3],
                    )
                )
                start += 0.5
            start += 2.0
        uri = f"method:_load_raw?timeline={self.timeline}"
        events.append(
            {
                "type": "Fmri",
                "filepath": uri,
                "start": 0,
                "frequency": 0.5,
                "duration": 20.0,
            }
        )
        return pd.DataFrame(events)
