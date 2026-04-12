# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import pandas as pd

from neuralset.data import BaseData

from ..download import Openneuro, Osf


class Grootswagers2022(BaseData):
    subject: str

    # study/class level attributes
    device: tp.ClassVar[str] = "Eeg"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds003825/versions/2.0.0"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{grootswagers2022human,
        title={Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams},
        author={Grootswagers, Tijl and Zhou, Ivy and Robinson, Amanda K and Hebart, Martin N and Carlson, Thomas A},
        journal={Scientific Data},
        volume={9},
        number={1},
        pages={3},
        year={2022},
        publisher={Nature Publishing Group UK London}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.18112/openneuro.ds003825.v1.1.0"
    licence: tp.ClassVar[str] = "CC-BY C0"
    description: tp.ClassVar[str] = "50 subjects watching still images in EEG."
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
    )

    _test_image_names: list[str] | None = None

    @classmethod
    def _download(cls, path: Path) -> None:
        # Download MEG data from OpenNeuro
        Openneuro(study="ds003825", dset_dir=path).download()
        # Download stimuli
        Osf("jum2f", path, folder="stimuli").download()  # type: ignore

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        """Returns a generator of all recordings"""

        for k in range(1, 51):
            folder = Path(path) / "download" / f"sub-{k:02}" / "eeg"
            assert folder.exists()
            yield cls(subject=str(k), path=Path(path))

    def _format_img_path(self, img_path: str) -> str:
        assert img_path.endswith(".jpg")
        assert img_path.startswith("stimuli")
        img_path = img_path.replace("\\", "/")
        assert img_path.count("/") == 2

        filename = Path(img_path).name
        category = Path(img_path).parent.name
        out = Path(self.path) / "prepare" / category / filename
        assert out.exists()
        return str(out)

    def _get_test_info(self) -> tuple[list[str], list[str]]:
        test_image_file = Path(self.path) / "download" / "test_images.csv"
        test_image_names = pd.read_csv(test_image_file, header=None)[0].tolist()
        test_categories = [name.split("/")[0] for name in test_image_names]
        return test_image_names, test_categories

    def _load_events(self) -> pd.DataFrame:
        subject = f"sub-{int(self.subject):02}"
        folder = Path(self.path) / "download" / subject / "eeg"
        events_file = folder / f"{subject}_task-rsvp_events.tsv"
        events = pd.read_csv(events_file, sep="\t")

        events["filepath"] = events.stim.apply(self._format_img_path)
        events["category"] = events.filepath.apply(lambda x: x.split("/")[-2])
        events["type"] = "Image"

        sfreq = 1e3
        events["start"] = events.onset / sfreq
        events.duration /= sfreq

        events["caption"] = events.filepath.apply(
            lambda x: " ".join(Path(x).stem.split("_")[:-1])
        ).apply(lambda s: re.sub(r"\d", "", s))

        # Identify test images (called "validation images" in the paper)
        test_image_names, test_categories = self._get_test_info()
        events["split"] = "train"
        image_names = events.filepath.apply(lambda x: "/".join(x.split("/")[-2:]))
        events.loc[image_names.isin(test_image_names), "split"] = "test"
        events["is_test_category"] = events.category.isin(test_categories)
        events["stem"] = events.filepath.apply(lambda x: Path(x).stem)

        # Add shared THINGS filepaths to images
        shared_things_path = (Path(self.path) / ".." / "THINGS-images").resolve(
            strict=False
        )
        if shared_things_path.exists():
            events["shared_filepath"] = (
                str(shared_things_path)
                + "/"
                + events.category
                + "/"
                + events.stem
                + ".jpg"
            )

        # Only keep useful columns
        events = events[
            [
                "filepath",
                "type",
                "start",
                "duration",
                "category",
                "caption",
                "split",
                "is_test_category",
                "stem",
                "shared_filepath",
            ]
        ]

        # Add raw Eeg event
        folder = Path(self.path) / "download" / subject / "eeg"
        raw_fname = folder / f"{subject}_task-rsvp_eeg.vhdr"
        # NOTE 1: Data was referenced online to Cz, which has not been recorded
        # NOTE 2: Subjects 1-48 have 63 channels; subject 49-50 have 127
        eeg = {"filepath": raw_fname, "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])

        return events
