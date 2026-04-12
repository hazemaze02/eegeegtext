# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

import neuralset as ns

from ..data import BaseData
from ..utils import success_writer
from .utils import add_sentences

logger = logging.getLogger(__name__)
logger.propagate = False


class Accou2023(BaseData):
    # Timeline level
    session: str
    run: str

    # Study level
    url: tp.ClassVar[str] = "https://www.mdpi.com/2306-5729/9/8/94"
    task: tp.ClassVar[str] = "listeningActive"
    bibtex: tp.ClassVar[str] = "TODO"
    licence: tp.ClassVar[str] = "CC-BY"
    device: tp.ClassVar[str] = "Eeg"
    description: tp.ClassVar[
        str
    ] = """
    A Speech-evoked Auditory Repository of EEG, measured at KU Leuven,
    comprising 64-channel EEG recordings from 85 young individuals with normal hearing,
    each of whom listened to 90-150 minutes of natural speech
    https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND
    """
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("mne_bids>=0.12",)

    @classmethod
    def _download(cls, path: Path) -> None:
        assert (
            path.name == cls.study.lower()
        ), "Specify the exact study folder when downloading."

        with success_writer(path / "download") as already_done:
            if not already_done:
                raise ValueError(
                    "Data can be downloaded using the original repo's utility script: "
                    "https://github.com/exporl/auditory-eeg-dataset/tree/master/download_code"
                    "\n"
                    "Then, manually create 'download_success.txt'."
                )

        with success_writer(path / "unzip") as already_done:
            if not already_done:
                logger.info("Unzipping files.")
                unzip_files(path)

        with success_writer(path / "word_onsets") as already_done:
            if not already_done:
                logger.info("Getting word onsets.")
                get_word_onsets(path)

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        from mne_bids import BIDSPath

        path = Path(path)
        assert path.exists(), "run study.download() first"
        subject_file = path / "participants.tsv"
        subjects = pd.read_csv(subject_file, sep="\t")

        def get_subject_id(x):
            return x.split("-")[1]  # noqa

        subjects = subjects.participant_id.apply(get_subject_id).values
        runs = [f"{x:02d}" for x in range(1, 11)]
        session_names = ["shortstories", "varyingStories"]
        session_numbers = [f"{x:02d}" for x in range(1, 11)]
        sessions = [f"{x}{y}" for x, y in product(session_names, session_numbers)]

        for subject, session, run in product(subjects, sessions, runs):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=cls.task,
                run=run,
                root=path,
                datatype="eeg",
            )
            if subject in ["19", "20", "21", "22"]:  # restricted rights
                continue

            if not Path(str(bids_path)).exists():
                logger.info("No file available for %s", str(bids_path))
                continue

            yield cls(subject=subject, session=session, run=run, path=path)  # type: ignore

    def _get_bids_path(self) -> tp.Any:
        from mne_bids import BIDSPath

        return BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            run=self.run,
            root=Path(self.path),
            datatype="eeg",
        )

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # pylint: disable=unused-import,disable=unused-argument
        # "timeline" is not used here but the uri serves for cache naming so must be unique
        """avoid re-reading all the headers"""

        bids_path = self._get_bids_path()
        raw = mne.io.read_raw_bdf(bids_path)
        if "EXG1" in raw.ch_names:
            raw.drop_channels(
                ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
            )
        raw.rename_channels(lambda name: name.replace("P0", "PO"))
        raw.set_montage("standard_1005", on_missing="ignore")
        return raw

    def _load_events(self) -> pd.DataFrame:
        stim_file = str(self._get_bids_path()).replace("eeg.bdf", "events.tsv")

        sound = pd.read_csv(stim_file, sep="\t")
        sound["type"] = "Sound"
        sound.drop(
            columns=["trigger_file", "noise_file", "video_file", "snr"], inplace=True
        )
        sound.rename(columns={"onset": "start", "stim_file": "filepath"}, inplace=True)
        sound_event = sound.iloc[0]

        sound_filepath = self.path / sound_event.filepath.replace(
            "eeg/", "annotated_stimuli/"
        ).replace(".npz.gz", ".wav")
        sound.filepath = sound_filepath

        words_filepath = str(sound_filepath).replace(".wav", ".tsv")
        words = pd.read_csv(words_filepath, sep="\t")
        words.rename(columns={"word": "text", "end": "stop"}, inplace=True)
        words["type"] = "Word"
        words["condition"] = sound_event.condition
        words["start"] = words["start"] + sound_event.start
        words["stop"] = words["stop"] + sound_event.start
        words["duration"] = words["stop"] - words.start
        words["language"] = "dutch"
        # remove empty words
        words = words[words.text != ""]

        uri = f"method:_load_raw?timeline={self.timeline}"
        eeg = pd.DataFrame([{"type": "Eeg", "filepath": uri, "start": 0}])

        events = pd.concat([eeg, sound, words], ignore_index=True)
        events.sort_values("start", inplace=True)

        events = add_sentences(events)

        return events


def unzip_files(data_path: str | Path, overwrite: bool = False):
    import concurrent.futures
    import gzip
    import os
    import shutil

    def decompress_gz_file(gz_file_path, output_file_path):
        with gzip.open(gz_file_path, "rb") as f_in:
            with open(output_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info("Decompressed: %s -> %s", gz_file_path, output_file_path)

    def find_and_decompress(root_dir):
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(".bdf.gz"):
                        gz_file_path = os.path.join(root, file)
                        output_file_path = gz_file_path[
                            :-3
                        ]  # Remove '.gz' from the filename
                        if (
                            os.path.exists(output_file_path)
                            and os.path.getsize(output_file_path) > 0
                        ):
                            if overwrite:
                                logger.info("Overwriting %s", output_file_path)
                            else:
                                continue

                        futures.append(
                            executor.submit(
                                decompress_gz_file, gz_file_path, output_file_path
                            )
                        )

        for future in concurrent.futures.as_completed(futures):
            future.result()

    find_and_decompress(data_path)


def get_word_onsets(data_path: str | Path, overwrite: bool = False):
    import gzip
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    import whisperx  # type: ignore
    from scipy.io.wavfile import write
    from tqdm import tqdm

    _rate = 48_000

    data_path = Path(data_path)
    zipped_files = list((data_path / "stimuli" / "eeg").glob("*.npz.gz"))
    savedir = data_path / "annotated_stimuli/"
    savedir.mkdir(parents=True, exist_ok=True)

    def load_npz_from_gz(gz_filepath):
        with gzip.open(gz_filepath, "rb") as f_in:
            # Read decompressed data into memory
            decompressed_data = BytesIO(f_in.read())
            # Load .npz file from memory
            data = np.load(decompressed_data, allow_pickle=True)
            return data

    print("Loading whisperx models")
    device = "cuda"
    compute_type = "float16"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(language_code="nl", device=device)

    for zipped_file in tqdm(zipped_files):
        wav_filename = str(zipped_file).replace(".npz.gz", ".wav")
        wav_filename = savedir / Path(wav_filename).name  # type: ignore
        transcript_filename = str(zipped_file).replace(".npz.gz", ".tsv")
        transcript_filename = savedir / Path(transcript_filename).name  # type: ignore
        if transcript_filename.exists():  # type: ignore
            if overwrite:
                logger.info("Overwriting %s", transcript_filename)
            else:
                continue

        # Save audio file
        data = load_npz_from_gz(zipped_file)["audio"]
        data = np.int16(data * np.max(np.abs(data)) * 32767)
        write(wav_filename, _rate, data)

        logger.info("Writing transcript...")
        audio = whisperx.load_audio(str(wav_filename))
        result = model.transcribe(audio, batch_size=16)
        transcript = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        with open(transcript_filename, "w") as f:
            f.write("word\tstart\tend\tsequence_id\tsentence\n")
            for i, segment in enumerate(result["segments"]):
                sentence = segment["text"]
                sentence = sentence.replace('"', "")  # remove quotes for parsing
                for word in segment["words"]:
                    if "start" not in word:
                        continue  # FIXME: for some reason, some words are missing start/end (especially numbers)
                    word["word"] = word["word"].replace('"', "")
                    f.write(
                        f"{word['word']}\t{word['start']}\t{word['end']}\t{i}\t{sentence}\n"
                    )


if __name__ == "__main__":
    path = "/storage/datasets01/shared/studies/accou2023"
    cache = "/storage/users/sdascoli/cache/sentence_decoding"

    events = ns.data.StudyLoader(
        name="Accou2023", path=path, cache=cache, download=True
    ).build()
