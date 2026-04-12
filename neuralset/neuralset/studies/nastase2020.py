# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import re
import typing as tp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from neuralset.data import BaseData
from neuralset.events import Event

from .utils import add_sentences

EXCLUDE_TASKS = ("notthefallshortscram", "notthefalllongscram", "schema")

# https://www.nature.com/articles/s41597-021-01033-3
# All studies used a repetition time (TR) of 1.5 seconds"
TR = 1.5


class _ConcatArray:
    """Lazily concatenate two lists of memmap 1D arrays"""

    # TODO find less hacky way to mock niftis :s

    def __init__(self, left: tp.List[np.ndarray], right: tp.List[np.ndarray]) -> None:
        self.left = left
        self.right = right
        n_voxels = len(left[0]) + len(right[0])
        n_times = len(left)
        assert len(right) == n_times
        self.shape = (n_voxels, n_times)
        self.ndim = 2

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple):
            key = key, slice(None)
        voxels, times = key
        if voxels != Ellipsis:
            raise NotImplementedError

        sel = range(*times.indices(self.shape[1]))

        # Function to extract the necessary data from a list of arrays

        def extract_data(hemi):
            data = [h for t, h in enumerate(hemi) if t in sel]
            return np.array(data).T

        # Extract data from left and right
        left = extract_data(self.left)
        right = extract_data(self.right)

        # Concatenate and return the result
        return np.concatenate((left, right), axis=0)

    def __array__(self) -> np.ndarray:
        out = np.concatenate((np.array(self.left).T, np.array(self.right).T), axis=0)
        assert out.shape == self.shape
        return out


class _PseudoHeader:
    def get_zooms(self):
        return (1.0, 1.0, 1.0, TR)


class _PseudoNifti:
    def __init__(self, path: str) -> None:
        """The preprocessed data is stored as two gifti files,
        one for each hemisphere.
        This class makes it possible to read it as Nifti single file
        while preserving the memmapping
        """
        import nibabel

        # read gitfi files with mmemap

        def read(path: str) -> tp.List[np.ndarray]:
            nii = nibabel.load(path, mmap=True)
            return [i.data for i in nii.darrays]  # type: ignore

        left = read(path % "L")
        right = read(path % "R")
        self.dataobj = _ConcatArray(left, right)
        self.header = _PseudoHeader()
        self.shape = self.dataobj.shape

    def get_fdata(self):
        return np.array(self.dataobj)


class Nastase2020(BaseData):
    url: tp.ClassVar[str] = "https://www.nature.com/articles/s41597-021-01033-3"
    licence: tp.ClassVar[str] = "CC-BY 0"
    device: tp.ClassVar[str] = "Fmri"
    description: tp.ClassVar[
        str
    ] = """
    345 subjects; 891 functional scans; 27 stories; each subject listened to a different set of stories
    """

    story: str
    session: int
    condition: str
    left_path: str
    right_path: str
    wav_path: str
    stim_start_tr: int
    # excluded: bool  # some records are defined as excluded, currently not included

    # doi = "https://doi.org/10.6084/m9.figshare.14818587"
    # data_url = "http://datasets.datalad.org/?dir=/labs/hasson/narratives/"
    # tr = 1.5
    # space = 'surface'
    # modality = "audio"
    # language = "en"

    # TODO: Add download method
    @classmethod
    def _download(cls, path: Path) -> None:
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        """Returns a generator of all recordings"""
        # download, extract, organize
        # cls.download()
        # List all recordings: depends on study structure
        df = _get_task_df(path)
        for _, row in df.iterrows():
            assert Path(row.wav_file).exists()
            assert Path(row.gii_fpath_left).exists()
            assert Path(row.gii_fpath_right).exists()
            yield cls(
                subject=row.subject,
                story=row.task,
                session=row.run,
                condition=row.condition,
                path=path,
                left_path=str(row.gii_fpath_left),
                right_path=str(row.gii_fpath_right),
                wav_path=str(row.wav_file),
                stim_start_tr=row.stim_start_tr,
                # excluded=row.exclude_left or row.exclude_right,
            )

    def _load_raw(self, timeline: str) -> _PseudoNifti:
        assert timeline == self.timeline
        path = str(self.left_path).replace("_hemi-L", "_hemi-%s")
        return _PseudoNifti(path)

    def _load_events(self) -> pd.DataFrame:
        """
        task: audio task
        stim_start_tr: number of TR before audio starts
        """

        # Events
        words = _get_words(self.path, self.story)
        phonemes = _get_phones(self.path, self.story)
        words["text"] = words.word_raw
        words["language"] = "english"
        phonemes["text"] = phonemes.phone
        phonemes["language"] = "english"

        keep = ["type", "text", "start", "duration", "language"]
        words = words[keep + ["sequence_id"]]
        phonemes = phonemes[keep]

        # Get sentences
        for _, df in words.groupby("sequence_id"):
            sentence = " ".join(df.text.values)
            words.loc[df.index, "sentence"] = sentence

        # Sound event with corresponding Text
        event = {
            "type": "Sound",
            "start": 0.0,
            "filepath": str(self.wav_path),
            "timeline": "tmp",
        }
        event = Event.from_dict(event).to_dict()  # populates duration
        event.pop("timeline")
        textfp = Path(self.wav_path)
        textfp = (
            textfp.parent
            / "transcripts"
            / textfp.name.replace("_audio.wav", "_transcript.txt")
        )
        if not textfp.exists():
            raise RuntimeError(f"Missing transcript file {textfp}")
        text: tp.Dict[str, tp.Any] = {
            "type": "Text",
            "text": textfp.read_text(),
            "language": "english",
        }
        text.update({x: event[x] for x in ["start", "duration"]})
        sound = pd.DataFrame([event, text])

        # Realign with the beginning of stimulus
        delay = TR * self.stim_start_tr
        words["start"] += delay
        phonemes["start"] += delay
        sound["start"] += delay

        # add train/test/val splits
        words = add_sentences(words)

        # Fmri event
        uri = f"method:_load_raw?timeline={self.timeline}"
        duration = self._load_raw(self.timeline).shape[-1] * TR
        fmri = pd.DataFrame(
            [
                dict(
                    type="Fmri",
                    start=0,
                    filepath=uri,
                    duration=duration,
                    frequency=1.0 / TR,
                )
            ]
        )

        # concatenate
        events = pd.concat([fmri, sound, words, phonemes], ignore_index=True)
        events["condition"] = self.condition
        events = events.sort_values("start").reset_index()

        return events


def _load_subjects_info(path: Path) -> pd.DataFrame:
    # Load participants information
    fname = path / "participants.tsv"
    raw_subject_df = pd.read_csv(fname, sep="\t").astype(str)
    subjects_df = []
    for row in raw_subject_df.itertuples():
        for task, condition, comprehension_str in zip(
            str(row.task).split(","),
            str(row.condition).split(","),
            str(row.comprehension).split(","),
        ):
            if comprehension_str != "n/a":
                comprehension = float(comprehension_str)
                if "shapes" in task:
                    comprehension /= 10
            else:
                comprehension = np.nan
            if task.startswith("notthefall"):
                condition = task.split("notthefall")[1]
                audio_task = task
            elif task != "milkyway":
                # milkyway and nothefall have different conditions
                audio_task = task
            else:
                audio_task = task + condition

            subjects_df.append(
                {
                    "subject": row.participant_id,
                    "task": audio_task,
                    "bold_task": task,
                    "condition": condition,
                    "comprehension": comprehension,
                }
            )
    return pd.DataFrame(subjects_df)


def _get_task_df(
    path: Path | str, exclude: bool = True, one_run_only: bool = False
) -> pd.DataFrame:
    # Partitipants info (subject, task, condition etc.)
    path = Path(path)
    subjects_df = _load_subjects_info(path)

    bi_df = []
    for hemi in ["left", "right"]:
        # Get gii_files (+mark excluded files)
        files_df = _get_gii_files_info(path, subjects_df, space="fsaverage6", hemi=hemi)

        # Merge
        df = pd.merge(subjects_df, files_df, on=["subject", "bold_task"], how="left")

        # Remove non existing sessions
        df = df.dropna(subset=["gii_fname"])
        df = df.astype({"run": int})

        # Remove excluded task
        df["exclude"] = df["exclude"].astype(bool)
        if exclude:
            df = df.query("not exclude").copy()
        df = df.query("task not in @EXCLUDE_TASKS").copy()  # no wave files?

        # Remove second run (only one scan per subject, task, hemi, space)
        if one_run_only:
            df = (
                df.sort_values(["subject", "task", "run"])
                .groupby(["subject", "task", "condition"])
                .agg("first")
                .reset_index()
            )

        # Add wavefile
        wavs = [path / "stimuli" / f"{task}_audio.wav" for task in df.task.values]
        for wav in wavs:
            assert wav.is_file()

        df["wav_file"] = wavs

        # Add the start TR of the stimulus
        task_onsets = _get_task_onsets()
        df["stim_start_tr"] = [task_onsets[task] for task in df["task"]]
        bi_df.append(df)

    df_out = pd.merge(
        bi_df[0],
        bi_df[1],
        on=[
            "subject",
            "task",
            "condition",
            "run",
            "wav_file",
            "stim_start_tr",
            "bold_task",
        ],
        suffixes=("_left", "_right"),
    )
    # condition: whether soud is scarmbled, intact etc.
    # bold_task = fmri file name
    # task = wave file name
    # run = session, there should be only one if one_run_only=True
    # bold_task redundant to gii
    assert df_out.subject.nunique() == 321 if exclude else 328
    assert df_out.task.nunique() == 18
    assert df_out.bold_task.nunique() == 16
    return df_out


def _get_gii_files_info(
    path: Path, subjects_df: pd.DataFrame, space: str = "fsaverage6", hemi: str = "left"
) -> pd.DataFrame:
    # Get corresponding bold files
    files_dict: tp.Dict[str, tp.List[tp.Any]] = defaultdict(list)
    for row in subjects_df.itertuples():
        gii_fname = f"{row.subject}_task-{row.bold_task}_*"
        gii_fname += f"space-{space}_hemi-{hemi[0].upper()}_desc-clean.func.gii"
        gii_files = list(
            (path / "derivatives" / "afni-nosmooth" / str(row.subject) / "func").glob(
                gii_fname
            )
        )
        for file in gii_files:
            fname = file.name
            pattern = r"run-(\w*)_"  # noqa
            run_match = re.findall(pattern, fname)
            run = int(run_match[0]) if len(run_match) else 1
            files_dict["subject"].append(row.subject)
            files_dict["bold_task"].append(row.bold_task)
            files_dict["gii_fname"].append(fname)
            files_dict["gii_fpath"].append(str(file))
            files_dict["run"].append(run)
    files_df = pd.DataFrame(files_dict)
    files_df["run"] = files_df["run"].astype(int)

    # Check for excluded sessions
    with (path / "code" / "scan_exclude.json").open("r") as f:
        exclude_dic = json.load(f)
    exclude = []
    for row in files_df.itertuples():
        row_exclude = False
        if row.bold_task in exclude_dic:
            if row.subject in exclude_dic[row.bold_task]:
                for pattern in exclude_dic[row.bold_task][row.subject]:
                    if pattern in row.gii_fname:  # type: ignore
                        row_exclude = True
        exclude.append(row_exclude)
    files_df["exclude"] = exclude
    return files_df


def _get_phones(path: Path | str, task: str) -> pd.DataFrame:
    path = Path(path)
    json_name = path / "stimuli" / "gentle" / task / "align.json"
    with json_name.open("r") as f:
        dico = json.load(f)
    phones = []
    for v in dico["words"]:
        if "phones" in v:
            current = v["start"]
            for i, phone in enumerate(v["phones"]):
                phones.append(
                    {
                        "phone": phone["phone"],
                        "start": current,
                        "duration": phone["duration"],
                        "offset": current + phone["duration"],
                        "phone_id": i,
                        "word": i,
                        "type": "Phoneme",
                    }
                )
                current += phone["duration"]
    return pd.DataFrame(phones)


def _get_words(path: Path | str, story: str) -> pd.DataFrame:
    path = Path(path)
    gentle_path = path / "stimuli" / "gentle" / story
    stim_fname = gentle_path / "align.csv"
    text_fname = gentle_path / "transcript.txt"
    columns = ["word", "word_low", "onset", "offset"]
    words = pd.read_csv(stim_fname, names=columns)
    _preproc_stim(words, text_fname, lower=False)
    _fix_stimulus(words, story)
    words["word_pp"] = words["word_raw"]
    words["sequence_id"] = words["sequ_index"]

    # some onset / offset are missing => interpolate
    words[["onset", "offset"]] = words[["onset", "offset"]].interpolate()
    words["start"] = words["onset"]
    words["duration"] = words["offset"] - words["onset"]
    words["condition"] = "sentence"
    words["type"] = "Word"
    words["word"] = words["word_pp"]
    words["duration"] = 0.01
    words["text"] = words.word_pp  # FIXME what is this

    return words


def _preproc_stim(df: pd.DataFrame, text_fname: str | Path, lower: bool = False) -> None:
    text = Path(text_fname).read_text()

    text = _format_text(text, lower=lower)
    transcript_tokens = _space_tokenizer(text)
    gentle_tokens = _gentle_tokenizer(text)
    assert len(gentle_tokens) == len(df)

    spans = _match_transcript_tokens(transcript_tokens, gentle_tokens)
    assert len(spans) == len(gentle_tokens)

    tokens = [w[0] for w in spans]
    tokens = _format_tokens(tokens, lower=lower)

    # word raw
    df["word_raw"] = tokens

    # is_final_word
    begin_of_sentences_marks = [".", "!", "?"]
    df["is_eos"] = [np.any([k in i for k in begin_of_sentences_marks]) for i in tokens]

    # is_bos
    df["is_bos"] = np.roll(df["is_eos"], 1)

    # seq_id
    df["sequ_index"] = df["is_bos"].cumsum() - 1

    # wordpos_in_seq
    df["wordpos_in_seq"] = df.groupby("sequ_index").cumcount()

    # wordpos_in_stim
    df["wordpos_in_stim"] = np.arange(len(tokens))

    # seq_len
    df["seq_len"] = df.groupby("sequ_index")["word_raw"].transform(len)

    # end of file
    df["is_eof"] = [False] * (len(df) - 1) + [True]
    df["is_bof"] = [True] + [False] * (len(df) - 1)

    df["word_raw"] = df["word_raw"].fillna("")
    df["word"] = df["word"].fillna("")


def _get_task_onsets() -> tp.Dict[str, int]:
    start_tr = {}
    # Set onsets for some tasks
    for key in [
        "21styear",
        "milkywayoriginal",
        "milkywaysynonyms",
        "milkywayvodka",
        "prettymouth",
        "pieman",
        "schema",
    ]:
        start_tr[key] = 0
    for key in ["piemanpni", "bronx", "black", "forgot"]:
        start_tr[key] = 8
    for key in [
        "slumlordreach",
        "shapessocial",
        "shapesphysical",
        "sherlock",
        "merlin",
        "notthefallintact",
        "notthefallshortscram",
        "notthefalllongscram",
    ]:
        start_tr[key] = 3
    for key in ["lucy"]:
        start_tr[key] = 2  # 1 in events.tsv, 2 in paper
    for key in ["tunnel"]:
        start_tr[key] = 2
    return start_tr


def _fix_stimulus(
    stimulus: pd.DataFrame,
    task: str,
    tasks_with_issues: tp.Sequence[str] = ("notthefallintact", "prettymouth", "merlin"),
    new_starts: tp.Any = ([25.8], [21], [29, 29.15]),
) -> None:
    if task in tasks_with_issues:
        new_vals = new_starts[tasks_with_issues.index(task)]
        for i, val in enumerate(new_vals):
            stimulus.loc[stimulus.index[i], "onset"] = val
            stimulus.loc[stimulus.index[i], "offset"] = val + 0.1


def _format_text(text: str, lower: bool = True) -> str:
    text = text.replace("\n", " ")
    text = text.replace(" -- ", ". ")
    text = text.replace(" - ", ", ")
    text = text.replace("-", "-")
    text = text.replace(' "', ". ")
    text = text.replace(' "', ". ")
    text = text.replace('" ', ". ")
    text = text.replace('". ', ". ")
    text = text.replace('." ', ". ")
    text = text.replace("?. ", "? ")
    text = text.replace(",. ", ", ")
    text = text.replace("...", ". ")
    text = text.replace(".. ", ". ")
    text = text.replace(":", ". ")
    text = text.replace("…", ". ")
    text = text.replace("-", " ")
    text = text.replace("  ", " ")
    if lower:
        text = text.lower()
    return text


def _match_transcript_tokens(transcript_tokens, gentle_tokens):
    transcript_line = np.array([i[1] for i in transcript_tokens])  # begin of each word
    raw_words = []
    for _, start, end in gentle_tokens:
        middle = (start + end) / 2
        diff = (middle - transcript_line).copy()
        diff[diff < 0] = np.Inf
        matching_idx = np.argmin(diff).astype(int)
        raw_words.append(transcript_tokens[matching_idx])

    return raw_words


def _gentle_tokenizer(raw_sentence):
    seq = []
    for m in re.finditer(r"(\w|\’\w|\'\w)+", raw_sentence, re.UNICODE):
        start, end = m.span()
        word = m.group()
        seq.append((word, start, end))
    return seq


def _split_with_index(s, c=" "):
    p = 0
    for k, g in itertools.groupby(s, lambda x: x == c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q  # or p, q-1 if you are really sure you want that
        p = q


def _format_tokens(x, lower=False):
    x = np.array(x)
    fx = [_format_text(" " + xi + " ", lower=lower).strip() for xi in x.reshape(-1)]
    fx = np.array(fx).reshape(x.shape)
    return fx


def _space_tokenizer(text):
    return [(text[i:j], i, j) for i, j in _split_with_index(text, c=" ")]
