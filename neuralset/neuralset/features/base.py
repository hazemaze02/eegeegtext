# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from abc import abstractmethod

import numpy as np
import pandas as pd
import pydantic
import torch

import neuralset as ns

from ..base import Frequency as Frequency
from ..base import _Module
from ..events import Event
from ..helpers import EventTypesHelper

T = tp.TypeVar("T", bound=torch.Tensor | np.ndarray)
logger = logging.getLogger(__name__)


class _NoEvent(Event):
    """Only used for checking that it is overriden"""


class BaseFeature(_Module):
    """Base class for defining features value based on a name.
    The aggregation parameter defines how to merge the values of multiple events.
    """

    event_type: tp.ClassVar[tp.Type[Event]] = _NoEvent
    event_types: str | tp.Tuple[str, ...] = ""
    #
    # event_type can now be overriden by an actual attribute
    # event_types so as to be specified at runtime.
    # This attribute must be either str or tuple of str,
    # types are not allowed because they can't be easily dumped in yaml files
    # eg:
    # event_types: str | tp.Tuple[str] = ("Image", "Text")  # type: ignore
    #
    aggregation: tp.Literal[
        "single", "sum", "average", "first", "middle", "last", "cat", "stack", "trigger"
    ] = "single"
    _CLASSES: tp.ClassVar[tp.Dict[str, tp.Type["BaseFeature"]]] = {}
    _event_types_helper: EventTypesHelper

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # check params
        super().__init_subclass__()
        # add event requirements to the feature reqirements
        if not cls._can_be_instanciated():
            return
        model_fields: dict[str, pydantic.FieldInfo] = cls.model_fields  # type: ignore
        event_types: tp.Any = model_fields["event_types"].default  # type:ignore
        name = cls.__name__
        if "event_type" in model_fields:
            raise RuntimeError(
                f"In {name!r}, event_type cannot be a variable, use event_types instead"
            )
        if event_types:  # not empty string
            if cls.event_type != _NoEvent:
                raise RuntimeError(
                    "In {name!r}, either specify event_type or a default for event_types"
                )
            if not isinstance(event_types, str):
                is_tuple = isinstance(event_types, tuple)
                if not (is_tuple and all(isinstance(d, str) for d in event_types)):
                    msg = f"In {name!r}, event_types attribute must be a string "
                    msg += f"or tuple of string, got {event_types}"
                    raise TypeError(msg)
        else:
            event_types = cls.event_type
            if event_types == _NoEvent:
                msg = (
                    f"In {name!r}, either specify event_type or a default for event_types"
                )
                raise RuntimeError(msg)
            if not issubclass(event_types, Event):
                msg = f"In {name!r}, event_type must be a type, got {event_types}"
                raise RuntimeError(msg)
        type_helper = EventTypesHelper(event_types)
        for etype in type_helper.classes:
            cls.requirements = cls.requirements + etype.requirements
        BaseFeature._CLASSES[cls.__name__] = cls
        if "name" not in model_fields or model_fields["name"].default != name:  # type: ignore
            # unfortunately, this field can't be added dynamically so far :(
            # https://github.com/pydantic/pydantic/issues/1937
            indication = f"name: tp.Literal[{name!r}] = {name!r}"
            raise NotImplementedError(
                f"Feature {name} has incorret/missing name field, add:\n{indication}"
            )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        event_types: tp.Any = self.event_types
        if not event_types:
            event_types = self.event_type
        self._event_types_helper = EventTypesHelper(event_types)

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        # feature convention from inheriting cache uid exclusion list
        return ["aggregation"]

    def prepare(self, events: pd.DataFrame) -> None:
        """Prepare all features for faster loading"""

    @abstractmethod
    def _get(self, event: Event, start: float, duration: float) -> float | torch.Tensor:
        """Provides data associated with 1 event.
        Needs to be overriden by user.

        event: pd.Series describing the event, contains start and duration.
        start: the start of the segment in the same timeline as the event
        duration: the duration of the segment.
        """
        raise NotImplementedError

    @tp.final
    def __call__(
        self,
        events: tp.Any,  # too complex: pd.DataFrame | list | dict | ns.events.Event,
        start: float,
        duration: float,
        trigger: float | tp.Dict[str, tp.Any] | None = None,
    ) -> torch.Tensor:
        """events: the single event (dict | ns.events.Event) or the series
        of events (list of Events | pd.DataFrame) describing the events, each
        containing start and duration.
        start: the start of the segment in the same timeline as the event.
        duration: the duration of the segment.
        """
        _input_events = events

        # Check argument
        assert duration >= 0.0, f"{duration} must be >= 0."
        event_types = self._event_types_helper.classes
        if self.aggregation == "trigger":
            type_ = trigger.get("type", None) if isinstance(trigger, dict) else trigger
            t: tp.Any = trigger
            if type_ in Event._CLASSES:  # convert to event if possible
                t = Event.from_dict(trigger)
            if not isinstance(t, event_types):  # clear error message
                aggregation = self.aggregation
                name = self.__class__.__name__
                msg = f"Feature {name} has {aggregation=} but trigger is {t!r} (not {event_types})"
                raise ValueError(msg)
            events = [t]
        if not isinstance(events, list):  # avoid many checks
            if isinstance(events, Event):
                events = [events]
            elif isinstance(events, dict):
                events = [Event.from_dict(events)]
        if isinstance(events, pd.DataFrame):
            # filter only useful events
            subclasses = self._event_types_helper.names
            events = events[events.type.isin(subclasses)]
            # skip itertuple if only one/two event :) (pandas is slooow)
            num = len(events)
            iterable = (
                (events.iloc[k, :] for k in range(num))
                if num <= 2
                else events.itertuples()
            )
            events = [Event.from_dict(r) for r in iterable]
        else:
            events = [e for e in events if isinstance(e, event_types)]

        if not events:
            found_types = {type(e) for e in _input_events}
            raise ValueError(
                f"No {event_types} found in segment (types found: {found_types} "
                f"in {_input_events})"
            )

        # Extract value for each relevant event
        if self.aggregation == "single":
            assert (
                len(events) < 2
            ), f"Found {len(events)} events in the segment but expected only one. Use the aggregation parameter to merge multiple events."
            out = self._get(events[0], start, duration)
        elif self.aggregation in ("first", "trigger"):
            out = self._get(events[0], start, duration)
        elif self.aggregation == "last":
            out = self._get(events[-1], start, duration)
        elif self.aggregation == "middle":
            out = self._get(events[len(events) // 2], start, duration)
        else:

            def _check_none(value: torch.Tensor | float, event: Event):
                if value is None:  # provide explicit error message in case of None
                    msg = f"Failed to compute feature for event of type {type(event)}"
                    raise ValueError(msg)
                return value

            values = (
                _check_none(self._get(event, start, duration), event) for event in events
            )
            out = self._aggregation(values)
        return out  # type: ignore

    def _aggregation(self, elements: tp.Iterable[torch.Tensor | float]) -> torch.Tensor:
        """how to merge the embeddings of multiple events"""
        if self.aggregation == "sum":
            out = sum(elements)
        elif self.aggregation == "average":
            out = None
            total = 0
            # iterate to avoid instantiating all elements at once
            for element in elements:
                if out is None:  # use float64 to avoid overflow
                    if isinstance(element, torch.Tensor):
                        out = torch.zeros_like(element, dtype=torch.float64)
                    else:
                        out = 0 * torch.Tensor(element)
                total += 1
                out += element
            if out is None:
                raise RuntimeError("No elements to aggregate")
            out = (out / total).to(torch.float32)
        elif self.aggregation == "cat":
            out = torch.cat(list(elements), dim=-1)  # type: ignore
        elif self.aggregation == "stack":
            out = torch.stack(list(elements), dim=-1)  # type: ignore
        else:
            raise ValueError(f"Unknown aggregation mode {self.aggregation}")
        return torch.Tensor(out)

    def _events_from_dataframe(self, events: pd.DataFrame) -> tp.List[tp.Any]:
        # we're loosing type here :(
        classes = self._event_types_helper.names
        filtered = events.loc[events.type.isin(classes), :]
        return [Event.from_dict(row) for row in filtered.itertuples()]


class BaseDynamic(BaseFeature):
    frequency: float | tp.Literal["native"] = 0.0
    _frequency_override: float | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.frequency != "native" and self.frequency < 0.0:
            msg = f"{self.__class__.__name__}.frequency is neither 'native' nor >= 0 (got {self.frequency})."
            raise ValueError(msg)
        if not (self.frequency or isinstance(self, BaseStatic)):
            msg = f"{self.__class__.__name__}.frequency=0 is only allowed for static features (did you mean 'native'?)"
            raise ValueError(msg)

    def _events_from_dataframe(self, events: pd.DataFrame) -> tp.List[tp.Any]:
        events_ = super()._events_from_dataframe(events)
        # check and log input frequencies in the "native" case
        if self.frequency == "native" and events_ and hasattr(events_[0], "frequency"):
            freqs = set(e.frequency for e in events_)  # type: ignore
            cls = self.__class__.__name__
            if len(freqs) > 1:
                msg = f"frequency='native' in {cls} with several different frequencies: {freqs}"
                msg += "\n(all data will not be processing at the same frequency, "
                msg += "should you set the feature frequency?"
                logger.warning(msg)
            elif len(freqs) == 1:
                cls = self.__class__.__name__
                freq = list(freqs)[0]
                msg = f"Processing to native frequency in {cls}.prepare: {freq}Hz"
                logger.info(msg)
        return events_

    def _output_frequency(self, event: ns.events.BaseDataEvent) -> Frequency:
        """Selects the output frequency for the feature tensor
        use frequency_override if provided, else feature frequency if different from native, else the
        event frequency
        """
        selected = event.frequency
        if self._frequency_override is not None:
            selected = self._frequency_override
        elif self.frequency != "native":
            selected = self.frequency
        return Frequency(selected)

    @staticmethod
    def _get_slice(freq, start: float, duration: float, decim: int = 1) -> slice:
        """Safely create a time slice"""
        assert duration >= 0.0, f"duration must be >= 0., got {duration}"
        freq = Frequency(freq)
        out_start = freq.to_ind(start)
        out_duration = max(1, freq.to_ind(duration))
        return slice(
            max(0, out_start),
            max(0, out_start + out_duration),
            decim,
        )

    @staticmethod
    def _get_overlap(
        event_start: float,
        event_duration: float,
        segment_start: float,
        segment_duration: float,
    ) -> tp.Tuple[float, float]:
        """
        Computes the overlap times between two windows
        """

        segment_stop = segment_start + segment_duration
        event_stop = event_start + event_duration

        overlap_start = max(segment_start, event_start)
        overlap_stop = min(segment_stop, event_stop)

        overlap_duration = max(0, overlap_stop - overlap_start)

        return overlap_start, overlap_duration

    def _get_overlap_slice(
        self,
        freq: float,
        event_start: float,
        event_duration: float,
        segment_start: float,
        segment_duration: float,
    ) -> tp.Tuple[slice, slice]:
        """
        get the overlap between the event and the segment:

        event:     [   ...]
        segment:      [...     ]
        """
        # get overlap times
        overlap_start, overlap_duration = self._get_overlap(
            event_start, event_duration, segment_start, segment_duration
        )
        out_slice = self._get_slice(freq, overlap_start - segment_start, overlap_duration)

        feature_freq = Frequency(freq if self.frequency == "native" else self.frequency)
        n_samp = max(1, feature_freq.to_ind(segment_duration))
        if out_slice.stop == n_samp + 1:
            # off by one due to rounding
            overlap_duration -= min(0.5 / feature_freq, overlap_duration)
            overlap_duration = max(0.001 / feature_freq, overlap_duration)  # avoid 0
            out_slice = self._get_slice(
                freq, overlap_start - segment_start, overlap_duration
            )
            if out_slice.stop - n_samp not in (0, 1):
                msg = f"Wrong output slice computation for {freq=}, {event_start=}, {event_duration=}, "
                msg += f"{segment_start=}, {segment_duration=}\n"
                msg += f"({n_samp=} and {out_slice=}"
                raise RuntimeError(msg)
        event_slice = self._get_slice(freq, overlap_start - event_start, overlap_duration)
        if overlap_duration == 0:
            out_slice = slice(out_slice.start, out_slice.start, out_slice.step)
            event_slice = slice(event_slice.start, event_slice.start, event_slice.step)
        return out_slice, event_slice

    def _fill_slice(
        self,
        data: np.ndarray | torch.Tensor,
        event: ns.events.BaseDataEvent,
        start: float,
        duration: float,
    ) -> torch.Tensor:
        """
        routine to retrieve a specific segment of data:

        data: [.......]
        segment    [oooooo]
        out:       [..oooo]

        assumes that the data is already in the right frequency
        """
        freq = self._output_frequency(event)  # deals with native/specified frequency
        exp_size = freq.to_ind(event.duration)
        if abs(data.shape[-1] - exp_size) > 1 or not data.shape[-1]:
            # data should not be empty and should be consistent with freq and duration
            raise ValueError(
                f"Data has incorrect (last) dimension {data.shape} for duration "
                f"{event.duration} and frequency {freq} (expected {exp_size})"
            )

        # get overlap times
        out_slice, event_slice = self._get_overlap_slice(
            freq,
            event.start,
            event.duration,
            start,
            duration,
        )

        # initialize output
        shape = np.r_[data.shape[:-1], max(1, freq.to_ind(duration))]
        out = torch.zeros(*shape, dtype=torch.float32)
        assert out[..., out_slice].size(), "the segment is empty"
        if event_slice.stop == data.shape[-1] + 1:
            # rounding for last sample of data, let's repeat the final sample
            es = event_slice
            event_slice = list(range(es.start, es.stop, es.step))  # type: ignore
            if event_slice:  # for empty case
                event_slice[-1] -= 1  # type: ignore
        data = data[..., event_slice]
        # check that the slices have same length
        if isinstance(data, np.ndarray):
            # need to copy the data as the array needs to be writable for torch
            data = torch.from_numpy(np.array(data, copy=True))
        out[..., out_slice] = data.float()
        return out


class BaseStatic(BaseDynamic):
    duration: float | None = None  # overrides the duration of the event
    frequency: float = 0.0

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.duration is not None and not self.frequency:
            msg = f"{self.__class__.__name__}.frequency is not > 0 while duration is."
            raise ValueError(msg)

    @abstractmethod
    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        """retrieve the static embedding"""
        raise NotImplementedError

    def _get(self, event: ns.events.Event, start: float, duration: float) -> torch.Tensor:
        """expand a static embedding to a particular window"""
        # get word emdedding
        embedding = self.get_static(event)
        if self.frequency == 0.0:
            return embedding

        frequency = Frequency(self.frequency)
        # brodcast it to the right time
        n_times = max(1, frequency.to_ind(duration))
        shape = np.r_[embedding.shape, n_times]
        out = torch.zeros(list(shape), dtype=torch.float32)

        # If duration is parametrized, we use the event duration
        if self.duration is None:
            event_duration = event.duration
        else:
            event_duration = self.duration

        # Expand embedding over time
        sl = self._get_slice(frequency, event.start - start, event_duration)
        out[..., sl] = embedding[..., None]
        return out


class Pulse(BaseStatic):
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Event
    name: tp.Literal["Pulse"] = "Pulse"

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        return torch.ones(1, dtype=torch.float32)


class Stimulus(BaseStatic):
    """Static event which sets the value to `code`."""

    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Stimulus
    name: tp.Literal["Stimulus"] = "Stimulus"

    def get_static(self, event: ns.events.Stimulus) -> torch.Tensor:
        return torch.tensor(event.code).long()


class LabelEncoder(BaseStatic):
    """Encode a given field from an event, e.g. to be used as a label.

    Parameters
    ----------
    event_types :
        Type of event to apply this feature to.
    event_field :
        Field to encode from the event.
    return_one_hot :
        If True, return one-hot representation of the index. Otherwise, return an int in
        [0, n_unique_values - 1].
    """

    name: tp.Literal["LabelEncoder"] = "LabelEncoder"
    event_types: str = "BaseDataEvent"
    event_field: str
    return_one_hot: bool = False

    _label_to_ind: dict[str, int] = {}

    def prepare(self, events: pd.DataFrame) -> None:
        self._label_to_ind = {
            ind: i
            for i, ind in enumerate(
                sorted(
                    events.loc[
                        (events.type == self.event_types),
                        self.event_field,
                    ].unique()
                )
            )
        }
        if not self._label_to_ind:
            raise ValueError(f"No type {self.event_types} found in events.")

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        if not self._label_to_ind:
            raise ValueError(
                "Must call label_encoder.prepare(events) before using the feature."
            )
        label = torch.tensor([self._label_to_ind[getattr(event, self.event_field)]])
        if self.return_one_hot:
            label = torch.nn.functional.one_hot(
                label, num_classes=len(self._label_to_ind)
            )
        return label[0]  # Remove unused first dimension
