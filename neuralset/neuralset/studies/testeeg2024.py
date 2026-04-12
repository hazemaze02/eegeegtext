# Synthetic EEG study for local testing — no data download required.
# Generates random EEG signals paired with word events so the full
# sentence_decoding pipeline can be exercised without any real dataset.

import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..data import BaseData

# Fixed vocabulary: enough variety for the model to learn embeddings
WORDS = [
    "the", "cat", "sat", "on", "mat", "dog", "ran", "far", "big", "red",
    "sun", "set", "sky", "was", "blue", "wind", "blew", "tree", "fell",
    "down", "bird", "sang", "song", "sweet", "rain", "came", "cold", "night",
    "moon", "rose", "high", "star", "shone", "bright", "sea", "waves",
    "crashed", "shore", "children", "played", "laughed", "smiled", "ran",
    "fast", "slow", "hill", "green", "grass", "grew", "tall", "flowers",
    "bloomed", "spring", "river", "flowed", "gently", "stones", "path",
    "wound", "through", "forest", "dark", "light", "filtered", "leaves",
]

# Build sentences as lists of words (8-10 words each)
SENTENCES = [
    ["the", "cat", "sat", "on", "the", "big", "red", "mat"],
    ["the", "dog", "ran", "far", "across", "the", "green", "grass"],
    ["the", "sun", "set", "and", "the", "sky", "was", "blue"],
    ["the", "wind", "blew", "cold", "through", "the", "tall", "trees"],
    ["the", "bird", "sang", "a", "sweet", "song", "at", "dawn"],
    ["the", "rain", "came", "down", "on", "the", "cold", "night"],
    ["the", "moon", "rose", "high", "and", "the", "star", "shone"],
    ["bright", "waves", "crashed", "hard", "upon", "the", "rocky", "shore"],
    ["the", "children", "played", "and", "laughed", "on", "the", "hill"],
    ["the", "green", "grass", "grew", "tall", "near", "the", "river"],
    ["flowers", "bloomed", "in", "spring", "along", "the", "winding", "path"],
    ["the", "river", "flowed", "gently", "over", "smooth", "round", "stones"],
    ["light", "filtered", "down", "through", "the", "leaves", "of", "trees"],
    ["the", "forest", "was", "dark", "and", "deep", "and", "cold"],
    ["a", "small", "bird", "sang", "from", "the", "highest", "branch"],
    ["the", "dog", "sat", "by", "the", "fire", "warm", "and", "still"],
    ["the", "sun", "shone", "bright", "on", "the", "morning", "sea"],
    ["the", "wind", "carried", "the", "scent", "of", "distant", "rain"],
    ["children", "ran", "fast", "down", "the", "long", "green", "hill"],
    ["the", "moon", "cast", "soft", "light", "on", "the", "still", "lake"],
    ["waves", "broke", "slow", "and", "sweet", "on", "the", "warm", "shore"],
    ["the", "cat", "watched", "the", "bird", "from", "behind", "the", "tree"],
    ["dark", "clouds", "gathered", "over", "the", "cold", "grey", "sea"],
    ["the", "path", "wound", "through", "tall", "grass", "and", "bright", "flowers"],
    ["the", "river", "sang", "as", "it", "ran", "over", "the", "stones"],
]


class TestEeg2024(BaseData):
    """Synthetic EEG dataset for local pipeline testing.

    Generates random EEG signals with realistic word-event structure.
    No data download required — everything is generated on the fly and
    cached as .fif files in the provided path.
    """

    device: tp.ClassVar[str] = "Eeg"

    # EEG parameters
    N_CHANNELS: tp.ClassVar[int] = 64
    SFREQ: tp.ClassVar[float] = 128.0        # Hz
    WORD_DURATION: tp.ClassVar[float] = 0.35  # seconds per word
    ISI: tp.ClassVar[float] = 0.15            # inter-stimulus interval (seconds)
    N_SUBJECTS: tp.ClassVar[int] = 3

    @classmethod
    def _download(cls, path: Path) -> None:
        # No download needed — data is generated in _load_raw
        pass

    @classmethod
    def _iter_timelines(cls, path: tp.Union[str, Path]) -> tp.Iterator["TestEeg2024"]:
        path = Path(path)
        for subject_id in range(cls.N_SUBJECTS):
            yield cls(subject=str(subject_id), path=path)

    def _load_events(self) -> pd.DataFrame:
        """Build a DataFrame of word + EEG events for one subject/timeline."""
        rows = []
        t = 2.0  # start a bit into the recording

        for sent_idx, sentence in enumerate(SENTENCES):
            sent_text = " ".join(sentence)
            for word in sentence:
                rows.append(dict(
                    type="Word",
                    text=word,
                    start=t,
                    duration=self.WORD_DURATION,
                    sentence=sent_text,
                    sequence_id=sent_idx,
                ))
                t += self.WORD_DURATION + self.ISI
            t += 1.0  # gap between sentences

        # Add the EEG recording event (points to _load_raw via URI)
        rows.append(dict(
            type="Eeg",
            start=0.0,
            filepath=f"method:_load_raw?timeline={self.timeline}",
        ))

        return pd.DataFrame(rows)

    def _load_raw(self, timeline: str) -> mne.io.Raw:
        """Generate (or load cached) a fake EEG Raw object."""
        # Filename encodes "unique 10–20 names + montage" (older `sub-*-raw.fif` had duplicate labels).
        fif_path = Path(self.path) / f"sub-{self.subject}-std1020-raw.fif"

        # Names and positions from a standard montage so MNE can build layouts
        # (ChannelPositions uses mne.find_layout; synthetic Raw needs digitization).
        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = list(montage.ch_names)[: self.N_CHANNELS]

        if fif_path.exists():
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        else:
            fif_path.parent.mkdir(parents=True, exist_ok=True)

            # Total recording length: enough to cover all words + some padding
            total_duration = (
                len(SENTENCES) * (
                    len(SENTENCES[0]) * (self.WORD_DURATION + self.ISI) + 1.0
                ) + 10.0
            )
            n_times = int(total_duration * self.SFREQ)

            np.random.seed(int(self.subject) * 42)
            info = mne.create_info(ch_names, sfreq=self.SFREQ, ch_types="eeg")
            # Simulate pink-ish noise (more realistic than white noise)
            data = np.random.randn(self.N_CHANNELS, n_times) * 10e-6  # ~10 µV
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.save(fif_path, verbose=False)
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

        raw.set_montage(montage, match_case=False, on_missing="ignore")
        return raw
