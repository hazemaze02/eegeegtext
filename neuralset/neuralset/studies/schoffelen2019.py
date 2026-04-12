# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..data import BaseData
from ..utils import match_list
from .utils import add_sentences

logger = logging.getLogger(__name__)

# The following subjects have 2 separate MEG runs, which would require
# specific code to handle.
# 1115 has nans in unexpected places.
BAD_NUMS = [2011, 2036, 2062, 2063, 2076, 2084, 1006, 1014, 1090, 1115]
NO_SUBJECT = [1014, 1018, 1021, 1023, 1041, 1043, 1047, 1051, 1056]
NO_SUBJECT += [1060, 1067, 1082, 1091, 1096, 1112]
NO_SUBJECT += [2012, 2018, 2022, 2023, 2026, 2043, 2044, 2045, 2048]
NO_SUBJECT += [2054, 2060, 2074, 2081, 2082, 2087, 2093, 2100, 2107]
NO_SUBJECT += [2112, 2115, 2118, 2123]


class Schoffelen2019(BaseData):
    # Recording variables
    subject: str
    modality: tp.Literal["audio", "visual"]

    # Study variables
    url: tp.ClassVar[str] = (
        "https://data.donders.ru.nl/collections/di/dccn/DSC_3011220.01_297"
    )
    # paper_url = "https://www.nature.com/articles/s41597-019-0020-y"
    doi: tp.ClassVar[str] = "https://doi.org/10.1038/s41597-019-0020-y"
    licence: tp.ClassVar[str] = "Donders"
    device: tp.ClassVar[str] = "Meg"
    description: tp.ClassVar[str] = (
        "204 subjects listened or read context-less sentences."
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "scipy",
        "git+https://github.com/kylerbrown/textgrid#egg=textgrid",
    )

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        # Donders: dccn/DSC_3011220.01_297
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["Schoffelen2019"]:

        for num in itertools.chain(range(1001, 1118), range(2002, 2126)):
            if num in BAD_NUMS + NO_SUBJECT:
                continue  # incomplete data
            modality = "visual" if num < 2000 else "audio"
            subject_uid = f"sub-{modality[0].upper()}{num}"
            yield cls(path=path, subject=subject_uid, modality=modality)  # type: ignore

    def _meg_file(self) -> str:
        meg_folder = self.path / "download" / self.subject / "meg"  # type: ignore
        meg_files = list(meg_folder.glob("*.ds"))
        meg_files = [x for x in meg_files if "rest" not in x.name]
        if not meg_files:
            if not meg_folder.exists():
                raise RuntimeError(
                    f"No MEG folder for recording {self.subject} at path\n{meg_folder}"
                )
            raise RuntimeError(f"No MEG file for recording {self.subject}")
        return str(meg_files[-1])

    def _metadata_file(self) -> str:
        metadata_folder = self.path / "download" / "sourcedata" / "meg_task"  # type: ignore
        search_string = f"*{self.subject.replace('sub-', '')}*"
        metadata_files = list(metadata_folder.glob(search_string))
        if not metadata_files:
            raise RuntimeError(f"No metadata file for recording {self.subject}")
        assert len(metadata_files) == 1  # TODO CHECK
        return str(metadata_files[-1])

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        """Loads raw file from the dataset"""
        # having a separate function is helpful for mocking
        raw = mne.io.read_raw(self._meg_file(), preload=False)
        if raw.info["sfreq"] != 1200:
            raise RuntimeError(
                "Raw has an unexpected sample rate, breaking code assumptions"
            )
        picks = mne.pick_types(
            raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
        )[28 : (28 + 273)]
        raw.pick(picks)  # only keep data channels
        return raw

    def _load_events(self) -> pd.DataFrame:
        """Loads events from the dataset"""
        # having a separate function is helpful for mocking
        raw = mne.io.read_raw(self._meg_file(), preload=False)
        sfreq = raw.info["sfreq"]
        events = mne.find_events(raw, shortest_event=1)
        # Read Event data

        # list meg folders
        metadata = _read_log(self._metadata_file())

        # Align MEG and events
        metadata = _get_log_times(metadata, events, sfreq)
        # rename
        metadata = metadata.rename(
            columns=dict(
                start="offset", meg_time="start", stop="legacy_stop", condition="kind"
            )
        )

        events_df = metadata.drop(
            columns=[x for x in metadata.columns if x.startswith("legacy_")]
        )

        # Clean up events
        sel = events_df.kind.isin(["word", "phoneme", "sound"])
        events_df = events_df.loc[sel]
        events_df["type"] = events_df.kind.str.capitalize()
        events_df["condition"] = events_df.context  # sentence or word_list
        events_df["sequence_id"] = events_df.sequence_uid
        words = events_df.type == "Word"
        events_df.loc[words, "text"] = events_df.loc[words, "word"]
        valid_words = events_df.loc[words].text.apply(len) > 0  # remove empty words
        events_df = events_df[~words | valid_words]
        events_df["sentence"] = events_df["word_sequence"]

        # clean up columns
        columns = [
            "start",
            "duration",
            "type",
            "condition",
            "text",
            "word_index",
            "sentence",
            # "sequence_id", # NOTE: the sequence_id is not consistent across timelines, remove it
        ]
        # -- phonemes and sounds
        phonemes = events_df.type == "Phoneme"
        if any(phonemes):
            events_df.loc[phonemes, "text"] = events_df.loc[phonemes, "phoneme"]
            columns += ["phoneme", "phoneme_id", "filepath"]

        events_df = events_df[columns]
        events_df = add_sentences(events_df, column_to_group="sentence")

        # add raw event from method
        uri = f"method:_load_raw?timeline={self.timeline}"
        meg = {"type": "Meg", "filepath": uri, "start": 0}
        events_df = pd.concat([pd.DataFrame([meg]), events_df])

        events_df[["language", "modality"]] = "dutch", self.modality

        return events_df


def _process_log_block(block: str) -> tp.List[tp.Dict[str, tp.Any]]:
    """Parse a block of annotation log"""
    lines = block.split("\n")
    # find header line
    iterlines = enumerate(lines)
    ind, line = next(iterlines)
    while "Uncertainty" not in line:  # only present in the header
        ind, line = next(iterlines)
    # build header (Uncertainty is present twice and must be updated)
    headers = [x.replace(" ", "_") for x in line.split("\t")]
    replacements = iter(["time_uncertainty", "duration_uncertainty"])
    for k, name in enumerate(headers):
        if name == "Uncertainty":
            headers[k] = next(replacements)
    # build data
    data: tp.List[tp.Dict[str, tp.Any]] = []
    for line in lines[ind + 1 :]:
        if not line:
            continue
        line_dict = dict(zip(headers, line.split("\t")))
        # convert to seconds if it's a time/duration field
        line_dict = {x: _seconds_if_time(x, y) for x, y in line_dict.items()}
        data.append(line_dict)
    return data


def _seconds_if_time(key: str, val: str) -> tp.Any:
    """Converts time/duration field values to seconds (initially in 1e-4)"""
    if val.isnumeric() and any(z in key.lower() for z in ["time", "dur"]):
        return float(val) / 1e4
    return val


def _parse_log(log_fname: str):
    text = Path(log_fname).read_text()

    # Fixes broken inputs
    text = text.replace(".\n", ".")

    # used to avoid duplicates in some subjects
    # FIXME it's unclear to me why these subjects have duplicated logs
    text = text.split("Scenario -")[1]

    # file is made of two blocks
    data1, data2 = [_process_log_block(block) for block in text.split("\n\n\n")]
    df1 = pd.DataFrame(data1)

    # # the two dataframe are only synced on certains rows
    common_samples = ("Picture", "Sound", "Nothing")
    sel = df1["Event_Type"].apply(lambda x: x in common_samples)
    index = df1.loc[sel].index
    df2 = pd.DataFrame(data2, index=index)

    # remove duplicate
    duplicates = np.intersect1d(df1.keys(), df2.keys())
    for key in duplicates:
        assert (df1.loc[index, key] == df2[key].fillna("")).all()
        df2.pop(key)

    log = pd.concat((df1, df2), axis=1)
    return log


def _clean_log(log):
    # Relabel condition: only applies to sample where condition changes
    translate = dict(
        ZINNEN="sentence",
        WOORDEN="word_list",
        FIX="fix",
        QUESTION="question",
        Response="response",
        ISI="isi",
        blank="blank",
    )
    for key, value in translate.items():
        sel = log.Code.astype(str).str.contains(key)
        log.loc[sel, "condition"] = value
    log.loc[log.Code == "", "condition"] = "blank"

    # Annotate sequence idx and extend context to all trials
    start = 0
    block = 0
    context = "init"
    log["new_context"] = False
    query = 'condition in ("word_list", "sentence")'
    for row in log.query(query).itertuples():
        idx = row.Index
        log.loc[start:idx, "context"] = context
        log.loc[start:idx, "block"] = block
        log.loc[idx, "new_context"] = True
        context = row.condition
        block += 1
        start = idx
    log.loc[start:, "context"] = context
    log.loc[start:, "block"] = block

    # Format time
    log.loc[:, "Time"] = [0.0 if not isinstance(x, (int, float)) else x for x in log.Time]

    # Extract individual word
    log.loc[log.condition.isna(), "condition"] = "word"
    idx = log.condition == "word"
    words = log.Code.str.strip("0123456789 ")
    log.loc[idx, "word"] = words.loc[idx]
    sel = log.query('word=="" and condition=="word"').index
    log.loc[sel, "word"] = np.nan
    log.loc[log.word.isna() & (log.condition == "word"), "condition"] = "blank"
    log.loc[log.Code == "pause", "condition"] = "pause"
    log.columns = log.columns.str.lower()  # remove capitalization!
    log.loc[log.word == "PULSE MODE", "condition"] = "pulse"
    return log


def _add_word_sequence_and_position(log: pd.DataFrame) -> pd.DataFrame:
    """Add word_sequence (the sequence of words in the sentence/word list) and
    word_index (its position) for each event of the log
    """
    indices = log.loc[log.condition == "fix"].index.tolist()
    for ind1, ind2 in zip(indices, indices[1:] + [log.index[-1]]):
        sub = log.loc[ind1:ind2, :]
        is_word = sub.condition == "word"
        sequence = " ".join(sub.loc[is_word, :].word)
        if sequence:
            log.loc[ind1:ind2, "word_sequence"] = sequence
            log.loc[ind1:ind2, "word_index"] = np.maximum(0, np.cumsum(is_word) - 1)
    return log


def _add_sound_events(path: str, log: pd.DataFrame):
    # Extract wave fname from structure
    sel = log["event_type"] == "Sound"

    def get_fname(s):
        name = s.split("Start File ")[1]
        return str(Path(path) / "stimuli" / "audio_files" / f"EQ_Ramp_Int2_Int1LPF{name}")

    filepaths = log.loc[sel, "code"].apply(get_fname)
    log.loc[sel, "filepath"] = filepaths

    # add wave fname to audio onset
    sel = log.query("event_type == 'Sound'").index  # type: ignore
    log.loc[sel + 1, "filepath"] = log.loc[sel, "filepath"].values

    log.loc[sel, "condition"] = "sound_legacy"
    log.loc[sel + 1, "condition"] = "sound"
    # features without "task" tag set are ignored during training,
    # so we set this tag properly
    # TODO move this one level up and mark "audio" for all "word" and "events" conditions
    return log


def _add_sequence_uid(path: str, log: pd.DataFrame):
    """Add sequence uid to the metadata."""
    # some trials missed the last word
    max_char = 45
    sequence_uids = dict()  # type: ignore
    with open(Path(path) / "stimuli" / "stimuli.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find(" ")
            uid = int(line[:idx])
            sequence = line[idx + 1 :].replace("\n", "")
            sequence = sequence[:max_char].lower()
            assert sequence not in sequence_uids.keys()
            sequence_uids[sequence] = uid
            assert uid != 0, "uid should not be 0"

    def _map(sequence):
        if not isinstance(sequence, str):
            return None
        key = sequence[:max_char].lower()
        assert key in sequence_uids, key
        return sequence_uids[key]

    sequence_uid: tp.Any = log.word_sequence.map(_map)
    first_idx = (sequence_uid.isna()).argmin()  # return first non NaN
    assert not (
        sequence_uid.iloc[first_idx:].isna()
    ).any(), "NaNs should be only at start"
    sequence_uid.iloc[:first_idx] = sequence_uid.iloc[first_idx]
    log["sequence_uid"] = sequence_uid
    return log


def _map_phonemes_to_ids_internal(phonemes_list, phonemes_ids_dict):
    phonemes_ids = []

    for phoneme in phonemes_list:
        key = phoneme.name
        assert key in phonemes_ids_dict, f"{key} not in dict {phonemes_ids_dict}"
        phonemes_ids.append(phonemes_ids_dict[key])
    return phonemes_ids


def _map_phonemes_to_ids(phonemes_list):
    ph_dict = {
        "d": 0,
        "@": 1,
        "b": 2,
        "A": 3,
        "n": 4,
        "s": 5,
        "i": 6,
        "E": 7,
        "r": 8,
        "x": 9,
        "p": 10,
        "o:": 11,
        "y": 12,
        "l": 13,
        "E:": 14,
        "Ei": 15,
        "N": 16,
        "e:": 17,
        "O": 18,
        "m": 19,
        "t": 20,
        "I": 21,
        "G": 22,
        "w": 23,
        "k": 24,
        "h": 25,
        "v": 26,
        "j": 27,
        "a:": 28,
        "u": 29,
        "z": 30,
        "Y": 31,
        "f": 32,
        "9y": 33,
        "S": 34,
        "ui": 35,
        "Au": 36,
        "Z": 37,
        "9:": 38,
        "2:": 39,
        "g": 40,
        "J": 41,
        "O:": 42,
    }
    return _map_phonemes_to_ids_internal(phonemes_list, ph_dict)


def _tgrid_to_dict(fname: str) -> tp.List[tp.Dict[str, tp.Any]]:
    """Parse TextGrid Praat file and generates a dataframe containing both
    words and phonemes"""
    import textgrid  # type: ignore

    tgrid = textgrid.read_textgrid(fname)
    parts: tp.Dict[str, tp.Any] = {}
    for p in tgrid:
        if p.name != "" and p.name != "<p:>":  # Remove empty entries
            parts.setdefault(p.tier, []).append(p)

    # Separate orthographics, phonetics, and phonemes
    words = parts["ORT-MAU"]
    phonemes = parts["MAU"]
    phonemes_ids = _map_phonemes_to_ids(phonemes)
    assert len(phonemes) == len(phonemes_ids)

    # Def concatenate orthographics and phonetics
    rows: tp.List[tp.Dict[str, tp.Any]] = []
    for word_index, word in enumerate(words):
        rows.append(
            dict(
                event_type="word",
                start=word.start,
                stop=word.stop,
                word_index=word_index,
                word=word.name,
                modality="audio",
            )
        )

    # Add timing of individual phonemes
    starts = np.array([i["start"] for i in rows])
    # phonemes and starts are both ordered so this could be further optimized if need be
    for phoneme, ph_id in zip(phonemes, phonemes_ids):
        indices = np.where(phoneme.start < starts)[0]
        idx = indices[0] - 1 if indices.size else len(rows) - 1
        row = rows[idx]
        rows.append(
            dict(
                event_type="phoneme",
                start=phoneme.start + 1e-6,
                stop=phoneme.stop,
                word_index=row["word_index"],
                word=row["word"],
                phoneme=phoneme.name,
                phoneme_id=ph_id,
                modality="audio",
            )
        )
    # not sure why sorting is needed, but otherwise a sample is dropped
    rows.sort(key=lambda x: float(x["start"]))
    return rows


def _add_phonemes(path: str, log: pd.DataFrame) -> pd.DataFrame:
    """Add phonemes and word timing to the log of the auditory experiment"""

    # Add audio file name across dataframe
    file_ = np.nan
    prev_start = 0
    prev_stop = 0

    log["sequence_id"] = np.nan
    starts = np.where(log.word.apply(lambda x: "Start File" in str(x)))[0]
    stops = np.where(log.word.apply(lambda x: "End of file" in str(x)))[0]

    assert len(starts) == len(stops)

    for start, stop in zip(starts, stops):
        # set file to previous rows
        log.loc[slice(prev_start, prev_stop), "sequence_id"] = file_
        # update file name
        file_ = int(log.loc[start, "word"].split()[-1][:-4])
        prev_start, prev_stop = start, stop
    log.loc[slice(prev_start, prev_stop), "sequence_id"] = file_

    # For each audio file, add timing of words and phonemes
    starts = np.where(log.word == "Audio onset")[0]
    rows: tp.List[tp.Dict[str, tp.Any]] = []  # faster than appending on the fly
    for start in starts:
        row = log.loc[start, :]
        if not row.condition == "sound":  # should be used for SentenceWavFeature
            raise RuntimeError(f"Unexpected condition {row.condition}")

        fname = "EQ_Ramp_Int2_Int1LPF%.3i.TextGrid" % row.sequence_id
        content = _tgrid_to_dict(str(Path(path) / "derivatives" / "phonemes" / fname))
        for d in content:
            d.update(
                subject=row.subject,
                trial=row.trial,
                stim_type="sound",
                context=row.context,
                block=row.block,
                sequence_id=row.sequence_id,
                duration=d["stop"] - d["start"],
                filepath=row.filepath,
                time=row.time + d["start"],
            )  # audio onset
        log.loc[start, "start"] = 0
        duration = content[-1]["stop"]
        log.loc[start, "stop"] = duration
        log.loc[start, "duration"] = duration
        rows.extend(content)
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True, sort=False)

    # homogeneize names
    for condition in ("word", "phoneme"):
        idx = log.query("event_type == @condition").index
        log.loc[idx, "condition"] = condition

    # fix
    idx = log.query('word=="End of file"').index
    log.loc[idx, "condition"] = "end"
    idx = log.query('event_type=="Nothing" and condition=="word"').index
    log.loc[idx, "condition"] = "nothing"
    return log.sort_values("time")


def _read_log(log_fname: str) -> pd.DataFrame:
    # guess study pass from log file
    path = str(Path(log_fname).parent.parent.parent)
    assert path.endswith("/download")

    log = _parse_log(log_fname)
    log = _clean_log(log)
    if "MEG-MOUS-Aud" in log_fname:
        log = _add_sound_events(path, log)
        log = _add_phonemes(path, log)
    elif "MEG-MOUS-Vis" in log_fname:
        words = log.query('condition == "word"')
        # TODO check duration?
        log.loc[words.index, "modality"] = "visual"
    else:
        raise ValueError(f"Unknown log type: {log_fname}")
    log = _add_word_sequence_and_position(log)
    # try:
    log = _add_sequence_uid(path, log)
    # except Exception:
    #    print("failure", log_fname)
    #    raise
    assert len(log)
    return log


def _get_log_times(log: pd.DataFrame, events: np.ndarray, sfreq: float) -> pd.DataFrame:
    from scipy.stats import spearmanr

    last_sample = events[-1, 0]
    sel: tp.Union[np.ndarray, slice] = np.sort(
        np.r_[
            np.where(events[:, 2] == 20)[0],  # fixation
            np.where(events[:, 2] == 10)[0],  # context
        ]
    )
    common_megs = events[sel]
    common_logs = log.query('(new_context == True) or condition=="fix"')

    last_log = common_logs.time.values[0]
    last_meg = common_megs[0, 0]
    last_idx = 0

    # TODO FIXME match_list may be based on too few elements, and
    # generate random timings, hence the assert > 40 (chosen arbitrarily)
    # fix missing triggers with leventhstein distance
    fix_logs = common_logs.code.str.contains("FIX")
    fix_megs = common_megs[:, 2] == 20
    if len(fix_megs) < 40 or len(fix_logs) < 40:
        logger.warning(
            "CAUTION: match_list may be based on too few elements, and "
            "generate random timings"
        )
    assert len(fix_megs) > 1 and len(fix_logs) > 1
    idx_logs, idx_megs = match_list(fix_logs, fix_megs)

    time_logs = common_logs.iloc[idx_logs].time
    time_meg = events[idx_megs, 0] * sfreq
    r, _ = spearmanr(time_logs, time_meg)
    # check that there is a perfect correlation between the log and meg timings
    assert r > 0.9999
    common_megs = common_megs[idx_megs]
    common_logs = common_logs.iloc[idx_logs]

    assert len(common_megs) == len(common_logs)
    for common_meg, common_log in zip(common_megs, common_logs.itertuples()):
        idx = common_log.Index
        if common_meg[2] == 20:
            assert common_log.condition == "fix"
        else:
            assert common_log.condition in ("sentence", "word_list")

        log.loc[idx, "meg_time"] = common_meg[0] / sfreq

        sel = slice(last_idx + 1, idx)
        times = log.loc[sel, "time"] - last_log + last_meg / sfreq
        assert np.all(np.isfinite(times.astype(float)))
        log.loc[sel, "meg_time"] = times

        last_log = common_log.time
        last_meg = common_meg[0]
        last_idx = idx  # type: ignore

        assert np.isfinite(last_log) * np.isfinite(last_meg)

    # last block
    sel = slice(last_idx + 1, None)
    times = log.loc[sel, "time"] - last_log + last_meg / sfreq
    log.loc[sel, "meg_time"] = times
    log["meg_sample"] = np.array(log.meg_time.fillna(0) * sfreq).astype(int)

    # Filter out events that are after the last MEG trigger
    n_out = np.sum(log.meg_sample > last_sample) + np.sum(log.meg_sample < 0)
    if n_out:
        logger.warning(
            f"CAUTION: {n_out} events occur after the last MEG trigger and will thus be removed"
        )

    log = log.query(f"meg_sample<={last_sample} and meg_sample>=0")

    return log


class SchoffelenRead2019(Schoffelen2019):
    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["SchoffelenRead2019"]:

        for num in range(1001, 1118):
            if num in BAD_NUMS + NO_SUBJECT:
                continue  # incomplete data
            modality = "visual"
            subject_uid = f"sub-{modality[0].upper()}{num}"
            yield cls(path=path, subject=subject_uid, modality=modality)  # type: ignore


class SchoffelenListen2019(Schoffelen2019):
    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["SchoffelenListen2019"]:

        for num in range(2002, 2126):
            if num in BAD_NUMS + NO_SUBJECT:
                continue  # incomplete data
            modality = "audio"
            subject_uid = f"sub-{modality[0].upper()}{num}"
            yield cls(path=path, subject=subject_uid, modality=modality)  # type: ignore
