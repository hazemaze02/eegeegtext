# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..data import BaseData
from ..utils import approx_match_samples, ignore_all, match_list
from .utils import add_sentences

mne.set_log_level(False)

CHAPTER_PATHS = [
    "ch1-3.wav",
    "ch4-6.wav",
    "ch7-9.wav",
    "ch10-12.wav",
    "ch13-14.wav",
    "ch15-19.wav",
    "ch20-22.wav",
    "ch23-25.wav",
    "ch26-27.wav",
]

# Handle the particular runs for which we need a higher tolerance
# To handle the default case, we'll use this in the later code
# abs_tol, max_missing = TOL_MISSING_DICT.get((subject, run), (10, 5))
TOL_MISSING_DICT = {
    (9, 6): (30, 5),  # Works
    (10, 6): (30, 5),  # Works
    (12, 5): (30, 5),  # Works
    (13, 3): (5, 30),  # Yes
    (13, 7): (5, 30),  # Yes
    (14, 9): (30, 5),  # Yes
    (21, 6): (200, 5),  # Yes
    (21, 8): (30, 5),  # No this is fked up: decided to toss sub21 run 8...
    (22, 4): (30, 5),  # Yes
    (33, 2): (40, 40),  # Yes (big shift so only 72% matched..)
    (39, 5): (45, 5),  # Yes
    (40, 2): (80, 5),  # Yes
    (41, 1): (40, 5),  # Yes
    (43, 4): (200, 5),  # Yes
    (43, 5): (110, 5),  # Yes
    (44, 9): (30, 5),
    (24, 2): (10, 20),
}

# Handle the runs that are particularly bad:
# - sub21 run 8
# - sub23 run 1-4
BAD_RUNS_LIST = [
    (21, 8),
    (23, 1),
    (23, 2),
    (23, 3),
    (23, 4),
]


class _Pallier2023Base(BaseData):
    # Timeline level
    session: str
    run: str

    # Study level
    version: tp.ClassVar[str] = "v3"  # change for cache update
    url: tp.ClassVar[str] = "TODO"
    bibtex: tp.ClassVar[str] = "TODO"
    licence: tp.ClassVar[str] = "TODO"
    device: tp.ClassVar[str] = "Meg"
    description: tp.ClassVar[
        str
    ] = """
    ~50 subjects listened or read Le Petit Prince in 9 runs.
    """
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "osfclient>=0.0.5",
        "mne_bids>=0.12",
        "Levenshtein>=0.23.0",
    )

    task: tp.ClassVar[str]

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        """Data are not yet on a public repo. They are available locally on jean-zay@idris.fr in
        `/gpfswork/rech/qtr/ulg98mt/data/LPP/` in `LPP_MEG_auditory_neuralset` and `LPP_MEG_visual_neuralset`
        You may also request them to christophe.pallier@inserm.fr"""
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        from mne_bids import BIDSPath

        path = Path(path)
        dl_dir = path / "download"  # TODO : download path
        assert dl_dir.exists(), "run study.download() first"
        subject_file = dl_dir / "participants.tsv"
        subjects = pd.read_csv(subject_file, sep="\t")

        def get_subject_id(x):
            return x.split("-")[1]  # noqa

        subjects = sorted(
            subjects.participant_id.apply(get_subject_id).values, key=int
        )  # type: ignore
        runs = ["{:02d}".format(x) for x in range(1, 10)]
        sessions = ["{:02d}".format(x) for x in range(1, 2)]  # 2 recording sessions
        for subject, session, run in product(subjects, sessions, runs):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=cls.task,
                run=run,
                root=dl_dir,
                datatype="meg",
            )

            # Currently skipping two dysfunctionnal subjects
            if ((int(str(subject)), int(run)) in BAD_RUNS_LIST) and cls.task == "listen":
                continue
            if not Path(str(bids_path)).exists():
                continue
            yield cls(subject=subject, session=session, run=run, path=dl_dir)  # type: ignore

    def _load_raw(self, timeline: str) -> mne.io.RawArray:  # type: ignore
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        from mne_bids import read_raw_bids

        bids_path = self._get_bids_path()
        with ignore_all():
            raw = read_raw_bids(bids_path)
        return raw

    def _get_bids_path(self):
        from mne_bids import BIDSPath

        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            run=self.run,
            root=self.path,
            datatype="meg",
        )
        return bids_path

    def _get_seq_id_path(self):
        return self.path / f"sourcedata/task-{self.task}_run-{self.run}_extra_info.tsv"

    def _get_syntax_path(self):
        return (
            self.path
            / f"sourcedata/stimuli/run{self.run}_v2_0.25_0.5-tokenized.syntax.txt"
        )

    def _get_word_info_path(self):
        return str(self._get_bids_path()).replace("meg.fif", "events.tsv")

    def _load_events(self) -> pd.DataFrame:
        """
        Redefine this method in the subclasses
        for both listen and read
        """
        raise NotImplementedError

    def _add_meg_and_ponctuated_context(
        self, raw: mne.io.RawArray, events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add the full text of the session to the events dataframe
        """
        # add raw event
        uri = f"method:_load_raw?timeline={self.timeline}"
        freq = raw.info["sfreq"]
        meg_text = [
            {
                "filepath": uri,
                "type": "Meg",
                "start": raw.first_samp / freq,
                "frequency": freq,
            }
        ]

        text_filepath = (
            Path(self.path)
            / "sourcedata"
            / "stimuli"
            / "txt_laser"
            / f"run{int(self.run)}.txt"
        )
        words = events.query('type=="Word"').copy()
        orig_text = Path(text_filepath).read_text()
        start = min(words.start)
        stop = max(words.start + words.duration)
        duration = stop - start

        dic_subsession = {
            "type": "Text",
            "text": orig_text.replace("\n\n", " ").replace("\n", " "),
            "start": start,
            "duration": duration,
            "language": "french",
        }
        meg_text.append(dic_subsession)
        events = pd.concat([pd.DataFrame(meg_text), events])
        return events


# # # # # actual studies # # # # #


class PallierListen2023(_Pallier2023Base):
    task: tp.ClassVar[str] = "listen"

    def _load_events(self) -> pd.DataFrame:
        """
        in this particular data, I'm transforming our original rich dataframe
        into mne use a Annotation class in order to save the whole thing into
        a *.fif file, At reading time, I'm converting it back to a DataFrame
        """

        error_msg_prefix = (
            f"subject {self.subject}, session {self.session}, run {self.run}\n"
        )

        raw = self._load_raw(self.timeline)
        # Get the start and stop triggers from STI101
        sound_triggers = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

        # extract annotations
        events = []
        for annot in raw.annotations:
            description = annot.pop("description")
            if "BAD_ACQ_SKIP" in description:
                continue
            event = eval(description)
            event["condition"] = "sentence"
            event["type"] = event.pop("kind").capitalize()
            event["start"] = annot["onset"]
            event["duration"] = annot["duration"]
            event["stop"] = annot["onset"] + annot["duration"]
            event["language"] = "french"
            events.append(event)

            # extract sound annotation
            sound_triggers = sound_triggers[sound_triggers[:, 2] == 1]  # get the triggers
            if np.sum(sound_triggers) != 2:
                warnings.warn(
                    f"No sound triggers found for subject {self.subject}, run {self.run}"
                )
            else:
                start, stop = sound_triggers[:, 0] / raw.info["sfreq"]
                events.append(
                    dict(
                        type="Sound",
                        start=start,
                        duration=stop - start,
                        filepath=Path(self.path)
                        / "sourcedata/stimuli/audio"
                        / CHAPTER_PATHS[int(self.run) - 1],
                    )
                )

        events_df = pd.DataFrame(events).rename(columns=dict(word="text"))

        # Remove empty words that were included in the metadata files...
        events_df = events_df[events_df["text"] != " "]

        metadata = pd.read_csv(self._get_seq_id_path())
        rows_events, rows_metadata = match_list(
            [str(word) for word in events_df["text"].values],
            [str(word) for word in metadata["word"].values],
        )

        assert len(rows_events) / len(events_df) > 0.95, (
            error_msg_prefix
            + f"only {len(rows_events) / len(events_df)} of the words were found in the metadata"
        )
        events_idx, metadata_idx = (
            events_df.index[rows_events],
            metadata.index[rows_metadata],
        )

        # Adding the information about sequence_id and n_closing
        events_df["word"] = events_df["text"]
        for col in ["sequence_id", "n_closing", "is_last_word", "pos"]:
            events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col]

        # add train/test/val splits
        events_df = add_sentences(events_df)  # TODO

        words = events_df.loc[events_df.type == "Word"]

        # Get the word triggers from STI008, as a step so we can get the offset
        word_triggers = mne.find_stim_steps(raw, stim_channel="STI008")
        # Offsets of the step function: allows us to match
        word_triggers = word_triggers[word_triggers[:, 2] == 0]

        # New match
        abs_tol, max_missing = TOL_MISSING_DICT.get(
            (int(self.subject), int(self.run)), (10, 5)
        )
        i, j = approx_match_samples(
            (words.start * 1000).tolist(),
            word_triggers[:, 0],
            abs_tol=abs_tol,
            max_missing=max_missing,
        )
        print(f"Found {len(i)/len(words)} of the words in the triggers")

        words = words.iloc[i, :]

        events_df.loc[:, "unaligned_start"] = events_df.loc[:, "start"]
        events_df.loc[words.index, "start"] = word_triggers[j, 0] / raw.info["sfreq"]

        ### Add punctuated context and raw event
        events_df = self._add_meg_and_ponctuated_context(raw, events_df)

        # sort by start
        events_df = events_df.sort_values(by="start").reset_index(drop=True)

        return events_df


class PallierRead2023(_Pallier2023Base):
    task: tp.ClassVar[str] = "read"

    def _load_events(self) -> pd.DataFrame:
        """
        in this particular data, I'm transforming our original rich dataframe
        into mne use a Annotation class in order to save the whole thing into
        a *.fif file, At reading time, I'm converting it back to a DataFrame

        We have to be careful when loading events, as:
        - In the MEG, the triggers in STI101 have a size of wlength (helps us realign the events with the MEG data)
        - This wlength has been calculated and sent to the MEG from the word as displayed, with ['"-],
        which is present under this form only in the events.tsv file (word column)

        """

        error_msg_prefix = (
            f"subject {self.subject}, session {self.session}, run {self.run}\n"
        )

        raw = self._load_raw(self.timeline)
        events = []
        words = pd.read_csv(self._get_word_info_path(), delimiter="\t")
        for _, row in words.iterrows():
            description = row["trial_type"]
            if "BAD_ACQ_SKIP" in description:
                continue
            event = eval(description)
            event["condition"] = "sentence"
            event["type"] = event.pop("kind").capitalize()
            event["start"] = row["onset"]
            event["duration"] = row["duration"]
            event["stop"] = row["onset"] + row["duration"]
            event["language"] = "french"
            event["text"] = row["word"]
            events.append(event)

        # The size of raw.annotations impacts the creation of the events_df: smaller than the number of events
        events_df = pd.DataFrame(events).rename(columns=dict(word="clean_text"))

        # TODO: this hack doesnt work as in read, the j and avais have been merged
        # It is thus needed to think about how to find again this information

        # Small data augmentation because some columns dont exist in the read metadata
        # metadata_listen = pd.read_csv(self.path / "sourcedata/task-listen_run-{self.run}_extra_info.tsv")
        # # Add to metadata the missing columns from the listen metadata: n_closing, is_last_word, pos, content_word
        # metadata = metadata.merge(metadata_listen[["word", "n_closing", "is_last_word", "pos", "content_word"]], on="word")

        word_triggers = mne.find_events(raw, stim_channel="STI101", shortest_event=1)
        words = events_df.loc[events_df.type == "Word"]
        words["wlength"] = words.text.apply(len)
        if word_triggers[:, 2].max() > 2048:
            word_triggers[:, 2] = (
                word_triggers[:, 2] - 2048
            )  # HACK because of a bug in the word_triggers for 2 subjects that have particularly high word_triggers

        # Matching the triggers wlength (with hyphens, dashes etc..) with the CORRECT metadata
        i, j = match_list(word_triggers[:, 2], words.wlength)
        words_recovered = len(j) / words.shape[0]

        assert words_recovered > 0.9, (
            error_msg_prefix
            + f"only {words_recovered} of the words were found in the word_triggers"
        )
        matched_word_indices = words.iloc[j].index

        # Create new type of events: missed words that were not found in the triggers
        events_df["unaligned_start"] = events_df["start"]
        missed_words = words[~words.index.isin(matched_word_indices)].copy()
        missed_words["type"] = "MissedWord"

        events_df.loc[matched_word_indices, "start"] = (
            word_triggers[i, 0] / raw.info["sfreq"]
        )

        # Drop the word events that were not found in the triggers
        false_indices = words[~words.index.isin(matched_word_indices)].index
        events_df.loc[false_indices, "start"] = np.nan
        events_df = events_df.dropna(subset=["start"])

        # Add the missed words to the events_df
        events_df = pd.concat([events_df, missed_words])

        # Match the events with the metadata
        metadata = pd.read_csv(self._get_seq_id_path())

        # Match with the metadata df that contains syntactic info, in order to append them later
        # Match it with the CLEAN text, as it is the one that is present in the extra_info
        rows_events, rows_metadata = match_list(
            [str(word) for word in events_df["clean_text"].values],
            [str(word) for word in metadata["word"].values],
        )

        assert len(rows_events) / len(events_df) > 0.8, (
            error_msg_prefix
            + f"only {len(rows_events) / len(events_df)} of the words were found in the metadata"
        )
        events_idx, metadata_idx = (
            events_df.index[rows_events],
            metadata.index[rows_metadata],
        )

        # Adding the information about sequence_id and n_closing
        events_df["word"] = events_df["text"]
        # for col in ["sequence_id", "n_closing", "is_last_word", "pos"]:
        for col in ["sequence_id"]:
            events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col]

        # add train/test/val splits
        events_df = add_sentences(events_df)  # TODO

        ### Add punctuated context and raw event
        events_df = self._add_meg_and_ponctuated_context(raw, events_df)

        # sort by start
        events_df = events_df.sort_values(by="start").reset_index(drop=True)

        return events_df
