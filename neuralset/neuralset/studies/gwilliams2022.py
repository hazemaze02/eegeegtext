# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralset.events import Event

from ..data import BaseData
from ..download import Osf
from .utils import add_sentences


class Gwilliams2022(BaseData):
    # Timeline level
    session: str
    story: str

    # Study level
    url: tp.ClassVar[str] = "https://www.biorxiv.org/content/10.1101/2020.04.04.025684v2"
    bibtex: tp.ClassVar[str] = "TODO"  # TODO
    licence: tp.ClassVar[str] = "CC-BY"
    device: tp.ClassVar[str] = "Meg"
    description: tp.ClassVar[
        str
    ] = """
    21 subjects listened to 4 stories in 2 x 1h identical sessions.
    """
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("osfclient>=0.0.5", "mne_bids>=0.12")

    @classmethod
    def _download(cls, path: Path) -> None:
        Osf("ag3kj", path).download()  # type: ignore
        Osf("h2tzn", path).download()  # type: ignore
        Osf("u5327", path).download()  # type: ignore

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        from mne_bids import BIDSPath

        path = Path(path)
        dl_dir = path / "download"
        assert dl_dir.exists(), "run study.download() first"
        subject_file = path / "download" / "participants.tsv"
        subjects = pd.read_csv(subject_file, sep="\t")

        def get_subject_id(x):
            return x.split("-")[1]  # noqa

        subjects = subjects.participant_id.apply(get_subject_id).values
        stories = [str(x) for x in range(4)]
        sessions = [str(x) for x in range(2)]  # 2 recording sessions
        for subject, session, story in product(subjects, sessions, stories):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=story,
                root=dl_dir,
                datatype="meg",
            )
            if not Path(str(bids_path)).exists():
                continue

            yield cls(subject=subject, session=session, story=story, path=path)  # type: ignore

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        """avoid re-reading all the headers"""
        from mne_bids import BIDSPath, read_raw_bids

        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.story,
            root=Path(self.path) / "download",
            datatype="meg",
        )
        raw = read_raw_bids(bids_path)
        return raw.copy()

    def _load_events(self) -> pd.DataFrame:
        """
        in this particular data, I'm transforming our original rich dataframe
        into mne use a Annotation class in order to save the whole thing into
        a *.fif file, At reading time, I'm converting it back to a DataFrame
        """

        # FIXME as bids is not a lazy read, we may want to
        # simply read the annotation manually
        raw = self._load_raw(self.timeline)

        # extract annotations
        events = list()
        for annot in raw.annotations:
            event = eval(annot.pop("description"))
            event["type"] = event.pop("kind").capitalize()
            event["start"] = annot["onset"]
            event["duration"] = annot["duration"]

            if event["type"] == "Sound":
                stem, _, ext = event["sound"].lower().rsplit(".", 2)
                event["filepath"] = Path(self.path) / "download" / (stem + "." + ext)

            event["language"] = "english"
            # Add text corresponding to the sound
            if event["type"] == "Sound":
                event["timeline"] = "#tmp#"
                event = Event.from_dict(event).to_dict()  # populates duration
                event.pop("timeline")
                sound_fp = Path(event["filepath"])
                *name, num_ext = sound_fp.with_suffix(".txt").name.split("_")
                textname = "_".join(list(name) + ["produced", num_ext])
                text_fp = sound_fp.parents[1] / "text_with_wordlists" / textname
                events.append(
                    {
                        "type": "Text",
                        # word column renamed to text, so let's hack it here
                        "word": text_fp.read_text("utf8")
                        .replace("\n\n", " ")
                        .replace("\n", " "),
                        "start": event["start"],
                        "duration": event["duration"],
                        "language": "english",
                    }
                )
            events.append(event)

        events_df = pd.DataFrame(events).rename(columns=dict(word="text"))
        pho = events_df.type == "Phoneme"
        events_df.loc[~pho].to_csv("bug.tsv", sep="\t")
        # propagate phoneme field to text for phonemes
        events_df.loc[pho, "text"] = events_df.loc[pho, "phoneme"]

        # add train/test/val splits
        events_df = add_sentences(events_df)

        # add raw event from methodevents.head
        uri = f"method:_load_raw?timeline={self.timeline}"

        # fill in the meg event totally as we have the raw open
        # (don't rely on Meg event to fill in freq and duration as it would require reloading)
        freq = float(raw.info["sfreq"])
        start = raw.first_samp / freq
        duration = raw.times[-1] - raw.times[0]
        meg = dict(
            type="Meg", filepath=uri, start=start, frequency=freq, duration=duration
        )
        events_df = pd.concat([pd.DataFrame([meg]), events_df], ignore_index=True)
        return events_df
