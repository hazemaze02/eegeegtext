# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
import typing as tp
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import mne
import numpy as np
import pandas as pd
import pydantic
from scipy.io import loadmat

from .. import utils
from ..data import BaseData
from .utils import add_sentences


class Broderick2019(BaseData):
    # Timeline level
    run_id: str

    # Study level
    url: tp.ClassVar[str] = (
        "http://datadryad.org/api/v2/datasets/doi%253A10.5061%252Fdryad.070jc/download"
    )
    bibtex: tp.ClassVar[str] = "TODO"  # TODO
    licence: tp.ClassVar[str] = "CC0 1.0"
    device: tp.ClassVar[str] = "Eeg"
    description: tp.ClassVar[str] = """"""
    language: tp.ClassVar[str] = "english"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()  # TODO
    _nlp: tp.Any = pydantic.PrivateAttr()

    @classmethod
    def _download(cls, path: Path) -> None:
        import gdown  # type: ignore

        dl_dir = path / "download"
        dl_dir.mkdir(exist_ok=True)
        url = "http://datadryad.org/api/v2/datasets/"
        url += "doi%253A10.5061%252Fdryad.070jc/download"
        zip_dset = dl_dir / "doi_10.5061_dryad.070jc__v3.zip"

        # download public files
        if not zip_dset.exists():
            print("Downloading Broderick_2019 dataset...")
            urlretrieve(url, zip_dset)

        # extract
        if not any([f.name == "N400.zip" for f in dl_dir.iterdir()]):
            print("Extracting Broderick_2019 dataset...")
            with ZipFile(str(dl_dir / zip_dset), "r") as zip_:
                zip_.extractall(str(dl_dir))
        dsets = [
            "Cocktail Party",
            "N400",
            "Natural Speech - Reverse",
            "Natural Speech",
            "Speech in Noise",
        ]
        for dset in dsets:
            subfolder = dl_dir / dset
            if not subfolder.exists():
                print(f"Extracting {dset}...")
                with ZipFile(str(subfolder) + ".zip", "r") as zip_:
                    zip_.extractall(str(dl_dir))

        # download private files FIXME TODO remove
        zip_private = dl_dir / "private.zip"
        if not zip_private.exists():
            print("Downloading Broderick_2019 private files...")
            url = "https://drive.google.com/u/0/uc?id="
            url += "1UAegPighc2t48CWpfBhjAdsZg1VlJ-mZ"
            gdown.download(url, str(zip_private), quiet=False)

        # extract private files
        folder_private = dl_dir / "private"
        if not folder_private.exists():
            print("Extracting Broderick_2019 private files...")
            with ZipFile(str(zip_private), "r") as zip_:
                zip_.extractall(dl_dir)

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        """Returns a generator of all recordings"""
        # download, extract, organize

        path = Path(path)
        dl_dir = path / "download"

        files = list((dl_dir / "Natural Speech" / "EEG").iterdir())
        subjects = [int(f.name.split("Subject")[1]) for f in files if "Subject" in f.name]
        subjects = sorted(subjects)

        for subject in subjects:
            for run_id in range(1, 21):
                recording = cls(subject=str(subject), run_id=str(run_id), path=path)
                yield recording

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        dl_dir = Path(self.path) / "download"
        eeg_fname = (
            dl_dir
            / "Natural Speech"
            / "EEG"
            / f"Subject{self.subject}"
            / f"Subject{self.subject}_Run{self.run_id}.mat"
        )
        mat = loadmat(str(eeg_fname))

        assert mat["fs"][0][0] == 128
        ch_types = ["eeg"] * 128
        # FIXME montage?
        montage = mne.channels.make_standard_montage("biosemi128")
        info = mne.create_info(montage.ch_names, 128.0, ch_types)
        eeg = mat["eegData"].T * 1e6
        assert len(eeg) == 128
        raw = mne.io.RawArray(eeg, info)
        raw.set_montage(montage)

        # TODO make mastoids EEG and add layout position
        info_mas = mne.create_info(["mastoids1", "mastoids2"], 128.0, ["misc", "misc"])
        mastoids = mne.io.RawArray(mat["mastoids"].T * 1e6, info_mas)
        raw.add_channels([mastoids])

        raw = raw.pick_types(meg=False, eeg=True, misc=False, eog=False, stim=False)
        return raw

    def _load_events(self) -> pd.DataFrame:
        # read and preprocess events from external log file
        # the files were shared manually and aligned with gentle
        events = self._get_events(self.run_id)

        events["language"] = self.language
        events = _extract_sentences(events)

        events = add_sentences(events)

        # add raw event from method
        uri = f"method:_load_raw?timeline={self.timeline}"
        eeg = {"type": "Eeg", "filepath": uri, "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])

        return events

    def _parse_json(self, run_id: str) -> pd.DataFrame:
        """parse json to flatten word and phoneme into a dataframe"""
        folder = Path(self.path) / "download"

        with open(folder / "private" / f"align{run_id}.json", "r") as f:
            align = json.load(f)

        meta = list()
        for entry in align["words"]:
            # for each event
            entry.pop("endOffset")
            entry.pop("startOffset")
            success = entry.pop("case") == "success"
            if not success:
                continue
            if entry["alignedWord"] == "<unk>":
                success = False
            entry["success"] = success

            # word level
            txt = entry.pop("word")
            entry["text"] = txt
            phones = entry.pop("phones")
            entry["phone"] = " ".join([k["phone"] for k in phones])
            entry["duration"] = entry["end"] - entry["start"]

            aligned = entry.pop("alignedWord")
            entry["aligned"] = aligned
            meta.append(entry)
            meta[-1]["type"] = "Word"

            # phoneme level
            start = entry["start"]
            for phone in phones:
                phone["start"] = start
                start += phone["duration"]
                phone["end"] = start
                phone["type"] = "Phoneme"
                phone["success"] = success
                phone["aligned"] = phone["phone"]
                phone["text"] = phone["phone"]
                meta.append(phone)
        # add audio entry
        wav = folder / "private" / f"audio{run_id}.wav"
        sound = dict(start=0, type="Sound", filepath=wav)

        df = pd.DataFrame([sound] + meta)
        df["duration"] = df["end"] - df["start"]

        return df

    def _parse_txt(self, run_id: str) -> pd.DataFrame:
        # read text
        txt_file = Path(self.path) / "download" / "private" / f"oldman_run{run_id}.txt"
        with open(txt_file, "r") as f:
            txt = f.read()

        # tokenize text
        doc = self._nlp(txt)

        # retrieve word and sentences
        sentences = []
        for sequence_id, sent in enumerate(doc.sents):
            seq_uid = str(sent)
            for word_index, word in enumerate(sent):
                word_ = re.sub(r"\W+", "", str(word))
                if not len(word_):
                    continue
                sentences.append(
                    dict(
                        text=word_,
                        original_text=word,
                        word_index=word_index,
                        sequence_id=sequence_id,
                        sequence_uid=seq_uid,
                    )
                )
        df = pd.DataFrame(sentences)
        return df

    def _get_events(self, run_id: str) -> pd.DataFrame:
        # lazy init
        if not hasattr(self, "_nlp"):
            self._nlp = utils.get_spacy_model(language="english")

        return self._process(run_id)

    def _process(self, run_id: str) -> pd.DataFrame:
        # read json file
        json = self._parse_json(run_id)

        # read text and parse with spacy
        text = self._parse_txt(run_id)

        # compare words in json and in text
        trans_words = json.query('type=="Word"')

        i, j = utils.match_list(trans_words.text.str.lower(), text.text.str.lower())
        assert len(i) > 450

        # add sequence information
        fields = ("sequence_id", "sequence_uid", "word_index")
        for k in fields:
            json.loc[trans_words.iloc[i].index, k] = text.iloc[j][k].values
        missed = np.setdiff1d(range(len(json)), trans_words.index[i])

        # fill-up missing information for phoneme and missed words
        prev = None
        indices = []
        for curr, _ in enumerate(json.sequence_id):
            if curr in missed:
                indices.append(json.index[curr])
            else:
                if len(indices) and prev is not None:
                    for k in fields:
                        json.loc[indices, k] = json.iloc[prev][k]
                    indices = []
                prev = curr

        json["condition"] = "sentence"
        for event_type in ("Word", "Phoneme"):
            idx = json.type == type
            json.loc[idx, event_type] = json.loc[idx].text

        return json


def _extract_sentences(events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sentences from a dataframe of events.
    """

    events_out = events.copy()
    is_word = events.type == "Word"
    words = events.loc[is_word]

    for _, d in words.groupby("sequence_id"):
        for uid in d.index:
            events_out.loc[uid, "sentence"] = " ".join(d.text.values)

    return events_out
