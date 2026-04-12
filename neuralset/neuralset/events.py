# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event handling classes and functions.

The class `Event` and its children (e.g. `Sound`, `Word`, etc.) define the
expected fields for each event type.
"""

import functools
import logging
import typing as tp
import urllib
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic

from .base import Frequency, StrCast, _Module
from .utils import ignore_all, warn_once

E = tp.TypeVar("E", bound="Event")
logger = logging.getLogger(__name__)


class Event(_Module):
    """Base class for all event types with the bare minimum common fields.

    If the event is instantiated with `from_dict()`, additional non-required
    fields that are provided will be ignored instead of causing an error.
    """

    start: float
    timeline: str
    duration: pydantic.NonNegativeFloat = 0.0
    _CLASSES: tp.ClassVar[tp.Dict[str, tp.Type["Event"]]] = {}
    type: tp.ClassVar[str] = "Event"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # register all events
        cls.type = cls.__name__
        Event._CLASSES[cls.__name__] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if pd.isna(self.start):
            raise ValueError(f"Start time needs to be provided for {self!r}")

    @classmethod
    def from_dict(cls: tp.Type[E], row: tp.Any) -> E:
        """Create event from dictionary/row/named-tuple while ignoring extra parameters."""
        if hasattr(row, "_asdict"):
            row = row._asdict()  # supports named tuples
        cls_ = cls._CLASSES[row["type"]]
        if not issubclass(cls_, cls):
            raise TypeError(f"{cls_} is not a subclass of {cls}")
        fs = set(cls_.model_fields)  # type: ignore
        kwargs = {k: v for k, v in row.items() if k in fs}
        for key, val in kwargs.items():
            # dataframe can replace empty strings or 0 values by nan
            # this is particularly the case for duration field, and language for Word
            if pd.isna(val):
                field = cls_.model_fields[key]  # type: ignore
                default = field.default
                null_default = not field.is_required() and (
                    not default or default is None
                )
                cases = [x for cls in (str, float, int) for x in (cls, cls | None)]
                if null_default and field.annotation in cases:
                    kwargs[key] = field.default
        try:
            out = cls_(**kwargs)
        except Exception as e:
            logger.warning(
                "Event.from_dict parsing failed for input %s\nmapped to %s\n with error: %s)",
                row.to_string() if hasattr(row, "to_string") else row,
                kwargs,
                e,
            )
            raise
        return out

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        # avoid Path in exports
        out = {x: str(y) if isinstance(y, Path) else y for x, y in self}
        out["type"] = self.type
        return out

    @property
    def stop(self) -> float:
        return self.start + self.duration


class BaseDataEvent(Event):
    """A base class for events who's data needs to be read from a file."""

    filepath: Path | str = ""
    frequency: float = 0
    _read_method: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.filepath:
            raise ValueError("A filepath must be provided")
        # check whether file actually points to register
        self._set_read_method()
        fp = str(self.filepath)
        self.filepath = fp
        if ":" not in str(fp):  # deactivate check
            # make sure to store file as a string in dataframe
            if not Path(fp).exists():
                warn_once(f"file missing: {fp}")

    def _set_read_method(self) -> None:
        try:
            if getattr(self, "_read_method", None) is not None:
                return
        except TypeError:  # pydantic bugs with private attr before model_post_init
            pass  # https://github.com/pydantic/pydantic/issues/9098
        tag = "method:"
        fp = str(self.filepath)
        if not fp.startswith(tag):
            self._read_method = self._read
            return
        # Store read method for reuse in subprocesses (where TIMELINES may not be filled)
        # avoid circular import:
        from .data import TIMELINES  # pylint: disable=import-outside-toplevel

        components = urllib.parse.urlparse(fp)
        assert components.netloc == ""
        assert components.params == ""
        assert components.fragment == ""
        # use a specific loader
        inst = TIMELINES[self.timeline]
        kwargs = dict(urllib.parse.parse_qsl(components.query, strict_parsing=True))
        self._read_method = functools.partial(getattr(inst, components.path), **kwargs)

    def __hash__(self) -> int:
        """required for lru_cache"""
        return hash(self.to_dict())

    def __eq__(self, other: tp.Any) -> bool:
        """required for lru_cache"""
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        return False

    def read(self) -> tp.Any:
        self._set_read_method()
        return self._read_method()

    @abstractmethod
    def _read(self) -> tp.Any:
        return

    def _missing_duration_or_frequency(self) -> bool:
        return any(not x or pd.isna(x) for x in [self.duration, self.frequency])


class Sound(BaseDataEvent):
    """Event corresponding to an audio Meg saved as a WAV file."""

    offset: pydantic.NonNegativeFloat = 0.0  # offset of the start within the sound file

    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("soundfile",)
    # soundfile (or PySoundFile?) or sox or sox_io may be needed as a backend
    # https://pytorch.org/audio/0.7.0/backend.html

    def model_post_init(self, log__: tp.Any) -> None:
        import soundfile

        if self._missing_duration_or_frequency():
            if not Path(self.filepath).exists():
                raise ValueError(f"Sound filepath does not exist: {self.filepath}")
            try:
                info = soundfile.info(str(self.filepath))
            except RuntimeError:
                if Path(self.filepath).exists():
                    logger.warning("A backend (soundfile, sox_io?) may be missing")
                raise
            if self.offset:
                raise RuntimeError(
                    "offset is provided while duration and/or frequency is missing"
                )
            self.duration = info.duration
            self.frequency = Frequency(info.samplerate)
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import soundfile
        import torch

        sr = Frequency(self.frequency)
        offset = sr.to_ind(self.offset)
        num = sr.to_ind(self.duration)
        fp = str(self.filepath)
        wav = soundfile.read(fp, start=offset, frames=num)[0]
        out = torch.Tensor(wav)
        if out.ndim == 1:
            out = out[None, :]
        return out

    def _split(
        self, timepoints: tp.List[float], min_duration: float | None = None
    ) -> tp.List["Sound"]:
        """Provided n ordered timepoints to split a the Sound event, returns
        the n + 1 corresponding Sound events corresponding to the sections

        Note
        ----
        timepoints are relative to the Sound event and not the absolute time in
        the sound file
        """
        # keep only timepoints that are within the sound duration
        timepoints = [t for t in timepoints if 0 < t < self.duration]
        timepoints = sorted(set(timepoints))
        if min_duration:
            delta_before = np.diff(timepoints, prepend=0)
            delta_after = np.diff(timepoints, append=self.duration)
            timepoints = [
                t
                for t, db, da in zip(timepoints, delta_before, delta_after)
                if db >= min_duration and da >= min_duration
            ]
        timepoints.append(self.duration)

        start = 0.0
        data = dict(self)
        cls = self.__class__
        events = []
        for stop in list(timepoints):
            if start >= stop:
                raise ValueError(
                    f"Timepoints should be strictly increasing (got {start} and {stop})"
                )
            data.update(
                start=self.start + start,
                duration=stop - start,
                offset=self.offset + start,
            )
            events.append(cls(**data))
            start = stop
        return events


class Image(BaseDataEvent):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("pillow>=9.2.0",)
    # Multiple captions for the same image should be '\n'-separated
    caption: str = ""

    def _read(self) -> tp.Any:
        # pylint: disable=import-outside-toplevel
        import PIL.Image  # noqa

        return PIL.Image.open(self.filepath).convert("RGB")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.duration <= 0:
            logger.info("Image event has null duration and will be ignored.")


class Video(BaseDataEvent):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("moviepy>=2.0.0",)

    def model_post_init(self, log__: tp.Any) -> None:
        if self._missing_duration_or_frequency():
            video = self.read()
            self.duration = video.duration
            self.frequency = Frequency(video.fps)
            video.close()
        super().model_post_init(log__)

    def _read(self) -> None:
        from moviepy import VideoFileClip  # noqa

        if not Path(self.filepath).exists():
            raise ValueError(f"Missing video file {self.filepath}")

        return VideoFileClip(str(self.filepath))


class BaseText(Event):
    """
    Base class for text events.
    """

    language: str = ""
    text: str = pydantic.Field("", min_length=1)


class Text(BaseText):
    """Possibly multi-sentence text"""

    language: str = ""
    text: str = pydantic.Field(..., min_length=1)


class Sentence(BaseText):
    """
    Sentence event.
    """


class Word(BaseText):
    """
    Word event.
    """

    # can be populated by data enhancers
    context: str = ""  # eg causal context
    sentence: str = ""  # sentence containing the word
    sentence_char: int | None = None  # character in the sentence


class Phoneme(BaseText):
    """"""


class Button(Text):
    """"""


class Motor(Event):
    """Event corresponding to a motor behavior."""


class Stimulus(Event):
    """General event corresponding to the presentation of a stimulus.

    As opposed to e.g., `ns.events.Image` and `ns.events.Sound` which point to an actual stimulus
    (e.g., image or sound) shown to participants, `ns.events.Stimulus` only registers a `code` (or
    "trigger" value) that is mapped to an event.

    See neuralset.studies.mnseample2013 for an example.
    """

    code: int = -100  # Default value ignored by CrossEntropyLoss (`ignore_index`)
    description: str = ""


class EyeState(Event):
    """Event of eye state. Can be 'open' or 'closed'.

    See neuralset.studies.babayan2019 for an example.
    """

    state: tp.Literal["closed", "open"]


class Artifact(Event):
    """An event corresponding to an artifact / noise event.
    Can be 'eyem', 'musc', 'chew', 'shiv', and 'elec'.
    - 'eyem': eye movement
    - 'musc': muscle artifact
    - 'chew': chewing
    - 'shiv': shivering
    - 'elpp': electrode artifact (electrode pop, electrode static, and lead artifacts)
    - 'artf': catch-all for artifact events
    See Hamid2020 ('tuar' sub_study) in neuralset.studies.lopez2017 for an example.
    """

    state: tp.Literal["eyem", "musc", "chew", "shiv", "elpp", "artf"]


class Seizure(Event):
    """An event corresponding to various forms of seizure.
    See Hamid2020 ('tuar' sub_study) in neuralset.studies.lopez2017 for an example.
    """

    state: tp.Literal[
        "bckg",  # Background
        "seiz",  # Seizure
        "gnsz",  # Generalized periodic epileptiform discharges
        "fnsz",  # Focal non-specific seizure
        "spsz",  # Simple partial seizure
        "cpsz",  # Complex partial seizure
        "absz",  # Absence seizure
        "tnsz",  # Tonic seizure
        "cnsz",  # Clonic seizure
        "tcsz",  # Tonic clonic seizure
        "atsz",  # Atonic seizure
        "mysz",  # Myoclonic Seizure
    ]


class EpileptiformActivity(Event):
    """An epileptic event.
    See Harati2015 ('tuev' sub_study) in neuralset.studies.harati2015 for an example.
    """

    state: tp.Literal[
        "spsw",  # Spike and/or sharp waves
        "gped",  # Generalized periodic epileptiform discharges
        "pled",  # Periodic lateralized epileptiform discharges
        "bckg",  # Background (no seizure)
    ]


class SleepStage(Event):
    """Stage of sleep following the American Association of Sleep Medicine manual [1]

    [1] "Berry, R., Quan, S. and Abreu, A. (2020) The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications, Version 2.6. American Academy of Sleep Medicine, Darien.
    """

    stage: tp.Literal[
        "W",  # Waking state
        "N1",  # Stage 1 of Non-Rapid Eye Movement (N-REM)
        "N2",  # Stage 2 of Non-Rapid Eye Movement (N-REM)
        "N3",  # Stage 3/4 of Non-Rapid Eye Movement (N-REM)
        "R",  # Rapid Eye Movement (REM)
    ]


class Meg(BaseDataEvent):
    """Brain Meg event"""

    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = self.subject
        if self._missing_duration_or_frequency():
            raw = self.read()
            self.duration = raw.times[-1] - raw.times[0]
            self.frequency = Frequency(raw.info["sfreq"])
            if raw.first_samp > 0 and not self.start:
                start = raw.first_samp / self.frequency
                msg = f"Meg event start for timeline {self.timeline} is 0 while "
                msg += f"raw.first_samp = {raw.first_samp} > 0\n"
                msg += f"(start should have been defined as raw.first_samp / raw.info['sfreq'] = {start})"
                raise ValueError(msg)
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import mne

        with ignore_all():
            return mne.io.read_raw(self.filepath)


class Eeg(Meg):
    """Brain Eeg event"""


class Emg(Meg):
    """Electromyography event"""


class Fnirs(Meg):
    """Brain Fnirs event"""

    def _read(self) -> tp.Any:
        import mne

        ext = Path(self.filepath).suffix
        with ignore_all():
            return {
                ".snirf": mne.io.read_raw_snirf,
                ".hdr": mne.io.read_raw_nirx,
                ".csv": mne.io.read_raw_hitachi,
                ".txt": mne.io.read_raw_boxy,
            }[ext](self.filepath)


class Fmri(BaseDataEvent):
    """Brain Fmri event"""

    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = str(self.subject)  # can be seen as int in dataframe
        if self._missing_duration_or_frequency():
            raise ValueError(
                "Duration and frequency must be provided for Fmri event: "
                "Don't rely on get_zooms as the header is sometimes unreliable.\n"
                f"Got: {self}"
            )
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import nibabel

        nii_img = nibabel.load(self.filepath, mmap=True)
        if nii_img.ndim not in (4, 2):  # type: ignore
            msg = f"{self.filepath} should be 2D or 4D with time the last dim."
            raise ValueError(msg)
        return nii_img
