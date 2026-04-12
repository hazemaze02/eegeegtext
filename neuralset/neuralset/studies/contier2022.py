# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import shutil
import typing as tp
import warnings
from functools import lru_cache
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from ..data import BaseData
from ..download import Openneuro, Osf
from ..utils import ignore_all

logger = logging.getLogger(__name__)


def _read_attributes_csv(path: str | Path, subject_id: str) -> pd.DataFrame:
    filename = f"sample_attributes_P{subject_id}.csv"
    csv = Path(path) / "download" / "sourcedata" / filename
    subject_events = pd.read_csv(csv, sep=",")
    return subject_events


def _load_events(
    path: str | Path, subject_id: str, session_id: int, run_id: int
) -> pd.DataFrame:
    subj_events = _read_attributes_csv(path, subject_id)

    # Add category and is_test_category information
    subj_events["stem"] = subj_events.image_path.apply(lambda x: Path(x).stem)
    subj_events["category"] = subj_events.stem.apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    subj_events["is_test_category"] = subj_events.category.isin(
        subj_events.loc[subj_events.trial_type == "test", "category"]
    )

    sel_session = subj_events.session_nr == session_id
    sel_run = subj_events.run_nr == run_id
    columns = ["trial_type", "image_on", "image_off", "image_path", "image_nr"]
    columns += ["things_image_nr", "stem", "category", "is_test_category"]
    events = subj_events.loc[sel_session & sel_run, columns]
    return events


@lru_cache
def _get_bids(subject: str, session: int, run: int, path: Path) -> mne.io.Raw:
    """mne bids is super slow, so let's cache it"""
    from mne_bids import BIDSPath  # pylint: disable=unused-import

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BIDSPath(
            subject=f"BIGMEG{subject}",
            session=f"{session:02d}",
            task="main",
            run=f"{run:02d}",
            datatype="meg",
            root=path / "download",
        )


class Contier2022(BaseData):
    session: int
    run: int

    # study/class level
    device: tp.ClassVar[str] = "Meg"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004212/versions/2.0.0"
    bibtex: tp.ClassVar[str] = "TODO"
    doi: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = "CC-BY C0"
    description: tp.ClassVar[str] = """4 subjects watching still images"""
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
    )

    @classmethod
    def _download(cls, path: Path, password: str | None = None) -> None:
        Osf("jum2f", path, folder="stimuli").download()  # type: ignore
        Openneuro("ds004212", path).download()  # type: ignore

        # Decompress
        parts = "A-C", "D-K", "L-Q", "R-S", "T-Z"
        prep_dir = path / "preprocessed" / "images"
        prep_dir.mkdir(parents=True, exist_ok=True)
        for part in parts:
            success = prep_dir / f"{part}_success.txt"
            if success.exists() or True:
                continue
            from pyunpack import Archive  # noqa

            if password is None:
                password = input("password (https://osf.io/srv7t):")

            img_dir = path / "download" / "stimuli" / "download" / "THINGS" / "Images"
            zip_file = img_dir / f"object_images_{part}.zip"
            print(f"Unzipping {zip_file.name}...")
            Archive(zip_file, password=password).extractall(prep_dir)

            # fix bad unzipping
            if parts == "L-Q":
                bad_dir = prep_dir / "object_images_L-Q"
                if bad_dir.exists():
                    for folder in bad_dir.iterdir():
                        if folder.is_file():
                            continue

                        target_dir = prep_dir / folder.name
                        target_dir.mkdir(exist_ok=True)
                        for file in folder.iterdir():
                            print(file)
                            shutil.move(str(file), str(target_dir / file.name))

            with open(success, "w") as f:
                f.write("done")

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        """Returns a generator of all recordings"""

        sub_ses_run = range(1, 5), range(1, 13), range(1, 11)
        for subject_id, session_id, run_id in product(*sub_ses_run):
            yield cls(  # type: ignore
                subject=str(subject_id),
                session=session_id,
                run=run_id,
                path=path,
            )

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        from mne_bids import BIDSPath, read_raw_bids  # noqa

        with ignore_all():
            # crazy mne warnings

            # FIXME use .ds directly
            # raw_fname = data_path / subject / session /'meg'/
            # f'{subject}_{session}_task-main_{run}_meg.ds'
            # raw = mne.io.read_raw_ctf(raw_fname)

            fname = _get_bids(self.subject, self.session, self.run, self.path)
            raw = read_raw_bids(fname)
            meg_types = ["MLC", "MLF", "MLO", "MLP", "MLT", "MRC", "MRF"]
            meg_types += ["MRO", "MRP", "MRT", "MZC", "MZF", "MZO", "MZP"]
            ch_types = dict()
            for c in raw.ch_names:
                if not any(c.startswith(t) for t in meg_types):
                    continue
                ch_types[c] = "mag"
            raw = raw.set_channel_types(ch_types)

            # FIXME TODO apparently weird layouts for
            # sub-BIGMEG2_ses-07_run-08
            # sub-BIGMEG3_ses-09_run-03
            # sub-BIGMEG3_ses-11_run-05

        return raw

    def _load_events(self) -> pd.DataFrame:
        raw_fname = _get_bids(self.subject, self.session, self.run, self.path)
        event_fname = str(raw_fname).replace("_meg.ds", "_events.tsv")
        raw_events = pd.read_csv(event_fname, sep="\t")

        events = _load_events(self.path, self.subject, self.session, self.run)

        assert len(events) == len(raw_events)

        # TODO FIXME 20 different values
        # np.array_equal(events.things_image_nr, raw_events.value)
        assert sum(events.things_image_nr.values != raw_events.value) < 21  # type: ignore

        # FIXME check why 150 ms offset between raw events and event.tsv?
        raw = self._load_raw(self.timeline)
        raw_events = mne.find_events(raw, stim_channel="UPPT001")
        assert len(raw_events) == (len(events) + 1)
        assert set(raw_events[:-1, 2]) == {64}
        assert raw_events[-1, 2] == 32

        starts = raw_events[:, 0] / raw.info["sfreq"]
        # diff = np.round(events.image_on - starts[:-1], 2)
        # if any(diff.unique()) > 0.16:
        #    print("UNEXPECTED EVENT ONSET: RAW AND TSV:", diff.unique())

        # TODO check onset and duration
        events["start"] = starts[:-1]  # events.image_on # FIXME check
        events["duration"] = events.image_off - events.image_on
        assert all(events.duration > 0.450)
        assert all(events.duration < 0.550)

        # specify image path
        img_dir = Path(self.path) / "prepare"

        def format_path(img_path):
            assert img_path.endswith(".jpg")
            filename = img_path.split("/")[-1]
            category = "_".join(filename.split("_")[:-1])
            return img_dir / category / filename

        events["filepath"] = events.image_path.apply(format_path)

        test = events.trial_type == "test"
        events["split"] = "train"
        events["valid"] = True
        events.loc[test, "split"] = "test"
        # FIXME what are those?
        valid = events.image_path.apply(lambda f: not f.startswith("images_catch_meg"))
        events.loc[~valid, "split"] = None
        events.loc[~valid, "valid"] = False
        check = events.filepath.apply(lambda x: x.exists())
        assert check.loc[valid].mean() > 0.95
        events.loc[~check, "split"] = None
        events.loc[~check, "valid"] = False

        # Add shared filepaths
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

        events["type"] = "Image"
        events["filepath"] = events.filepath.apply(str)
        events["caption"] = events.category.str.replace("_", " ").apply(
            lambda s: re.sub(r"\d", "", s)
        )
        invalid = int((~valid).sum())
        if invalid:
            logger.warning(
                "Removing %s invalid (catch images) events from contier2022", invalid
            )
            events = events.loc[valid, :]
        # add raw event from method
        uri = f"method:_load_raw?timeline={self.timeline}"
        meg = {"type": "Meg", "filepath": uri, "start": 0}
        events = pd.concat([pd.DataFrame([meg]), events])
        return events
