# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from neuralset import utils

from ..data import BaseData
from ..download import Donders
from ..utils import match_list
from .utils import add_sentences

TASK = "compr"  # (useless/internal) constant in the study


class Armeni2022(BaseData):
    # Timeline level
    session: str

    # Study level
    url: tp.ClassVar[str] = "https://www.nature.com/articles/s41597-022-01382-7#Sec16"
    licence: tp.ClassVar[str] = "https://data.donders.ru.nl/doc/dua/RU-DI-HD-1.0.html?12"
    device: tp.ClassVar[str] = "Meg"
    description: tp.ClassVar[
        str
    ] = """
    3 subjects listening to 10 hours of sherlock holmes.
    """
    version: tp.ClassVar[str] = "v3"

    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "mne_bids>=0.12",
        "Levenshtein>=0.23.0",
        "spacy>=3.5.4",
    )

    @classmethod
    def _download(cls, path: Path) -> None:
        Donders(
            study="armeni2022", study_id="DSC_3011085.05_995_v1", dset_dir=path
        ).download()
        cls.fix_bad_files(path)

    @classmethod
    def fix_bad_files(cls, path: str | Path) -> None:
        # fix bad files
        def fix_file(sub, ses, source, target):
            fname = path / "download" / sub / ses / f"{sub}_{ses}_scans.tsv"
            txt = fname.read_text()
            if source in txt:
                txt = txt.replace(source, target)
                fname.write_text(txt)

        fix_file("sub-002", "ses-003", "n/a\nt", "n/a\n")
        fix_file("sub-003", "ses-009", ".ds n/a", ".ds\tn/a")

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
        sessions = [str(x).zfill(3) for x in range(1, 11)]  # 10 recording sessions

        for subject, session in product(subjects, sessions):
            if subject == "003" and session == "008":
                # 2 sound events aligned with words,
                # followed by 4 sound events with words that do not match at all
                # see: https://github.com/fairinternal/brainai/pull/797
                continue
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=TASK,
                root=dl_dir,
                datatype="meg",
            )
            if not Path(str(bids_path)).exists():
                print(f"Skipping {bids_path}")
                continue

            yield cls(path=path, subject=str(subject), session=session)

    def _load_events(self) -> pd.DataFrame:
        from mne_bids import BIDSPath

        events = self._read_annotations(self.subject, self.session)
        events["sequence_id"] = None
        events["condition"] = None

        sounds = events.query('type=="Sound"')
        stops = sounds.start.shift(-1, fill_value=np.inf)
        for start, stop, filepath in zip(sounds.start, stops, sounds.filepath):
            sound_id = int(filepath.split("_")[-1].strip(".wav"))
            assert sound_id in range(1, 10)
            doc = self._read_text(self.session, sound_id)

            # select words corresponding to sound_id
            df = (events.start >= start) & (events.start < stop)
            words = events.loc[df].query('type=="Word"').copy()
            words_annots = words.text.str.lower()

            # match the two
            words_nlp = [w.text.lower() for w in doc]
            idx, jdx = match_list(words_nlp, words_annots)

            sentence_uid = [str(doc[i].sent) for i in idx]
            events.loc[words.iloc[jdx].index, "sequence_id"] = sentence_uid

        # fill missing
        prev = None
        for event in events.itertuples():
            if event.type in ["Sound", "Meg"]:
                continue
            if not isinstance(event.sequence_id, str):
                assert isinstance(prev, str)
                events.loc[event.Index, "sequence_id"] = prev
            else:
                prev = event.sequence_id

        words = events.query('type!="Sound"')
        events["sequence_id"] = np.cumsum(
            words.sequence_id != words.sequence_id.shift(1, fill_value=1)
        )

        events["condition"] = None
        events.loc[words.index, "condition"] = "sentence"

        events = add_sentences(events)

        # add raw event from method
        raw_fname = str(
            BIDSPath(
                subject=self.subject,
                session=self.session,
                task=TASK,
                root=Path(self.path) / "download",
                datatype="meg",
            )
        )

        sounds = events.query('type=="Sound"')

        all_subsessions_text = {}
        all_subsessions_start_stop = {}

        for start, stop, filepath in zip(sounds.start, stops, sounds.filepath):
            sound_id = int(filepath.split("_")[-1].strip(".wav"))
            assert sound_id in range(1, 10)
            text_read = self._read_text_with_text_and_doc(self.session, sound_id)
            all_subsessions_text[sound_id] = text_read

            df = (events.start >= start) & (events.start < stop)
            words = events.loc[df].query('type=="Word"').copy()
            start = min(words.start)
            stop = max(words.start + words.duration)
            duration = stop - start
            all_subsessions_start_stop[sound_id] = [start, duration]

        meg_text = [
            {"type": "Meg", "filepath": raw_fname, "start": 0},
        ]

        for subsession in all_subsessions_text.keys():
            dic_subsession = {
                "type": "Text",
                "text": all_subsessions_text[subsession]
                .replace("\n\n", " ")
                .replace("\n", " "),
                "start": all_subsessions_start_stop[subsession][0],
                "duration": all_subsessions_start_stop[subsession][1],
                "language": "english",
            }
            meg_text.append(dic_subsession)

        events = pd.concat([pd.DataFrame(meg_text), events], ignore_index=True)
        return events

    def _read_text_with_text_and_doc(self, session, sound_id):
        path = self.path / "download" / "stimuli"
        text_fname = path / f"{session[1:]}_{sound_id}.txt"
        text = text_fname.read_text()
        return text

    def _read_text(self, session, sound_id):
        nlp = utils.get_spacy_model(language="english")
        path = self.path / "download" / "stimuli"
        text_fname = path / f"{session[1:]}_{sound_id}.txt"
        text = text_fname.read_text()
        doc = nlp(text.replace("\n", " "))
        return doc

    def _read_annotations(self, subject, session):
        assert len(session) == 3
        path = self.path / "download"
        # read events
        stim_file = (
            path
            / f"sub-{subject}"
            / f"ses-{session}"
            / "meg"
            / f"sub-{subject}_ses-{session}_task-{TASK}_events.tsv"
        )
        df = pd.read_csv(stim_file, sep="\t")

        # preproc annotations
        df = df.rename(columns=dict(onset="start"))

        sounds = df.query('type=="wav_onset" and value != "100"').copy()
        sounds["type"] = "Sound"

        def get_soundid(sound_id):
            sound_id = int(sound_id) // 10
            assert sound_id in range(1, 10)
            return str(path / "stimuli" / f"{session[1:]}_{sound_id}.wav")

        sounds["filepath"] = sounds["value"].apply(get_soundid)

        words = df.type.str.contains("word")
        phonemes = df.type.str.contains("phoneme")
        words = df.loc[words].query("value!='sp'").copy()
        phonemes = df.loc[phonemes].query("value!='sp'").copy()

        words["type"] = "Word"
        words["text"] = words["value"]
        phonemes["type"] = "Phoneme"
        phonemes["text"] = phonemes["value"]
        phonemes["start"] += 1e-6

        # concatenate
        out = pd.concat([sounds, words, phonemes], ignore_index=True)
        out.sort_values("start", inplace=True)
        phs = out.query('type=="Phoneme"')
        out.loc[phs.index, "start"] += 1e-6

        keep = ["start", "duration", "text", "type", "filepath"]
        return out[keep]
