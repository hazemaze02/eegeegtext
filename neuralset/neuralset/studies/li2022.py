# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import zipfile
from pathlib import Path

import pandas as pd

from neuralset.data import BaseData
from neuralset.download import Openneuro
from neuralset.events import Event


class Li2022(BaseData):
    # Timeline level
    run: int
    lang: tp.Literal["FR", "CN", "EN"]  # useful for filtering

    # Study level
    version: tp.ClassVar[str] = "v3"
    url: tp.ClassVar[str] = "https://www.nature.com/articles/s41597-022-01625-7"
    bibtex: tp.ClassVar[
        str
    ] = """@dataset{ds003643:2.0.5,
author = {Jixing Li and John Hale and Christophe Pallier},
title = {"Le Petit Prince: A multilingual fMRI corpus using ecological stimuli"},
year = {2024},
doi = {doi:10.18112/openneuro.ds003643.v2.0.5},
publisher = {OpenNeuro}
}"""
    licence: tp.ClassVar[str] = "CC0"
    device: tp.ClassVar[str] = "Fmri"
    description: tp.ClassVar[
        str
    ] = """
    ~30 French ~50 English and ~30 Chinese subjects listened Le Petit Prince in 9 runs, inside a 3T fMRI.
    """
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("openneuro-py", "praat-textgrids")
    dataset_id: tp.ClassVar[str] = "ds003643"
    TR_FMRI_S: tp.ClassVar[float] = 2.0

    @classmethod
    def _download(cls, path: Path) -> None:
        """
        Data is available here:
        https://openneuro.org/datasets/ds003643/versions/2.0.5/
        Make sure to get the last version of the dataset.
        """
        dataset_id = "ds003643"
        if path.name.lower() != cls.__name__.lower():
            path = path / cls.__name__
        client = Openneuro(study=dataset_id, dset_dir=path)
        client.download()
        # TODO when available, automatize download of transcripts files

    @classmethod
    def _iter_timelines(cls, path: Path | str):
        path = Path(path)
        # loop across subjects
        folder = path
        if not folder.exists():
            raise ValueError(f"No folder {folder}")
        lang_subjects = [
            ("FR", range(1, 31)),
            ("EN", range(57, 116)),
            ("CN", range(1, 38)),
        ]
        en_missing = [60, 66, 71, 80, 85, 90, 102, 107, 111, 112]
        for lang, subject_indices in lang_subjects:
            for sub_index in subject_indices:
                if lang == "FR" and sub_index in [21, 27]:
                    continue  # Those subjects have been removed in the last version
                if lang == "EN" and sub_index in en_missing:
                    continue  # does not exist
                if lang == "CN" and sub_index in [12, 35]:
                    continue  # does not exist
                for run in range(1, 10):
                    sub = f"sub-{lang}{sub_index:03d}"
                    yield cls(subject=sub, lang=lang, run=run, path=path)  # type: ignore

    def _load_events(self) -> pd.DataFrame:
        """Load events"""
        import nibabel
        import textgrids as tg

        path = Path(self.path) / "download"
        # word events
        txt_grid = (
            path / f"annotation/{self.lang}/lpp{self.lang}_section{self.run}.TextGrid"
        )
        textgrid = tg.TextGrid(txt_grid)
        keys = list(textgrid)
        if len(keys) > 1:
            raise RuntimeError(f"Only one key should be in textgrid, got {keys}")
        # fixes to match text and annotations
        repl = {
            "three_hundred_twenty-five": "325",
            "six_hundred_twelve": "612",
            "one_thousand_nine_hundred_nine": "1909",
            "one_thousand_nine_hundred_twenty": "1920",
            "minster": "minister",
            'na\\i""ve': "naive",
            # french
            "coeur": "cœur",
            "oeil": "œil",
        }
        wordseq: tp.List[tp.Dict[str, tp.Any]] = []
        duplicated = "it the this that i i. they we he she you now and but so there five twenty phew good".split()
        for i in textgrid[keys[0]]:
            if i.text in ["", "#"]:  # new sentence
                continue
            # duplicated words happen a lot (when missing new sentence character), merge them
            if i.text in duplicated and wordseq:
                if wordseq[-1]["text"] == i.text:
                    wordseq[-1]["duration"] = i.xmax - wordseq[-1]["start"]
                    continue
            # add word
            text = (
                repl.get(i.text, i.text)
                .replace("\`", "'")
                .replace("«", "")
                .replace("»", "")
            )
            wordseq.append({"text": text, "start": i.xmin, "duration": i.xmax - i.xmin})

        events = pd.DataFrame(wordseq)
        events["type"] = "Word"
        # Example sound file:
        # task-lppFR_section_1.wav
        sep = "-" if self.lang == "EN" else "_"
        soundfile = path / f"stimuli/task-lpp{self.lang}_section{sep}{self.run}.wav"
        # TODO need to validate start time in some way, it may be incorrect:
        # "There was a trigger at the beginning of each section and a delay of 8 s (4 TRs)
        # between the trigger and onset of stimulus presentation for all three languages"
        event = {
            "type": "Sound",
            "start": 0,
            "filepath": str(soundfile),
            "timeline": "tmp",
        }
        event = Event.from_dict(event).to_dict()  # populates duration
        event.pop("timeline")
        events2 = [event]
        lang = self.lang.lower()
        textzip = Path(self.path) / "transcripts" / f"lpp_{lang}_text.zip"
        if not textzip.exists():
            raise RuntimeError(f"Transcripts file must be manually dumped in {textzip}")
        with zipfile.ZipFile(textzip) as archive:
            text = archive.read(
                f"lpp_{lang}_text/text_{lang}_run{self.run}.txt",
                pwd=b"lessentielestinvisiblepourlesyeux",
            ).decode("utf8")
        # text fixes
        if lang == "en":
            text = text.replace("did I have this sense", "did I have to have this sense")
            text = text.replace("my little price", "my little prince")
            asteroids = "3 2 5, 3 2 6, 3 2 7, 3 2 8, 3 2 9, and 3 3 0"
            if asteroids in text:
                replasteroids = asteroids
                for num, word in [
                    (0, "zero"),
                    (2, "two"),
                    (5, "five"),
                    (6, "six"),
                    (7, "seven"),
                    (8, "eight"),
                    (9, "nine"),
                ]:
                    replasteroids = replasteroids.replace(str(num), word)
                text = text.replace(asteroids, replasteroids)
            text = text.replace("street lamp", "streetlamp")
            text = text.replace("red faced", "redfaced")
        elif lang == "fr":
            for old, new in [("’", "'"), (" pusse ", " puisse ")]:
                text = text.replace(old, new)
        language = {"en": "english", "fr": "french", "cn": "chinese"}[lang]
        # end of text fixes
        events2.append({"type": "Text", "text": text, "language": language})
        # update text start/duration from sound start/duration
        events2[-1].update({x: events2[-2][x] for x in ["start", "duration"]})

        # add fmri event
        fmri_folder = path / "derivatives" / self.subject / "func"
        if not fmri_folder.exists():
            raise RuntimeError(f"Subject fmri folder does not exist: {fmri_folder}")
        files = sorted(fmri_folder.glob("*_space-MNIColin27_desc-preproc_bold.nii.gz"))
        if not len(files) == 9:
            raise RuntimeError(f"There should be 9 run files in {fmri_folder}")
        # the run in the filename may be different ("scanning issues or participants needing a break")
        # so this filename would have been incorrect:
        # f"{self.subject}_task-lpp{self.lang}_run-{self.run:02d}_space-MNIColin27_desc-preproc_bold.nii.gz"
        niifile = files[self.run - 1]
        nii: tp.Any = nibabel.load(niifile, mmap=True)
        freq = 1.0 / self.TR_FMRI_S
        dur = nii.shape[-1] / freq
        events2.append(
            dict(type="Fmri", start=0, filepath=niifile, frequency=freq, duration=dur)
        )
        out = pd.concat([events, pd.DataFrame(events2)], ignore_index=True)
        return out.reset_index(drop=True)
