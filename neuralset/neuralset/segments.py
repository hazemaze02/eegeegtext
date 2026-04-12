# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd

from .events import BaseDataEvent, Event
from .utils import warn_once

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Segment:
    """What gets out of a `ns.segments.list_segments(events, duration=1.)`.
    Fields:
    - df: pd.DataFrame (dataframe of events occuring in this time segment)
    - start: float (start time of the time segment)
    - duration: float (duration of the time segment)

    Additionally there is a lazily computed and cached "events" field
    as a list of Event instances, for faster feature processing
    """

    # adding a dict interface to this class confuses pytorch-lightning
    # so better avoid it and keep to a standard dataclass

    events: pd.DataFrame
    start: float
    duration: float
    _trigger: float | tp.Dict[str, tp.Any] | None = None
    _event_list: tp.List["Event"] | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    @property
    def event_list(self) -> tp.List["Event"]:
        """cached list of event instances for faster processing"""
        if self._event_list is None:
            # itertuple is so slow for small lengths
            e = self.events
            n = len(e)
            iterable = (e.iloc[k, :] for k in range(n)) if n <= 2 else e.itertuples()
            self._event_list = [
                Event.from_dict(r) for r in iterable if r.type in Event._CLASSES  # type: ignore
            ]
        return self._event_list

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def asdict(self) -> tp.Dict[str, tp.Any]:
        """Allows for feature(**segment.asdict())"""
        return {"events": self.events, "start": self.start, "duration": self.duration}


def _validate_event(event: pd.Series) -> dict[str, tp.Any]:
    """Validate event, i.e. check fields and values are as expected,
    and update it accordingly.

    This is done by instantiating an event object of the corresponding
    type, which carries out the validation, and then updating the input
    with the applied changes (if any).
    """
    # Check types are valid
    event_type = event["type"]
    lower = {x.lower() for x in Event._CLASSES}
    if event_type in Event._CLASSES:
        event_class = Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()

        # Add back fields that were ignored by the Event class
        # segment.update(asdict(event_obj))
        # Very slow, use dict updating instead
        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(f"Legacy uncapitalized event {event}")
    else:
        warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`neuralset.events`."
        )
        event_dict = {**event}

    return event_dict


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the DataFrame of events (not inplace).

    Returns
    -------
    pd.DataFrame
        DataFrame in which each row has been validated and updated.
    """
    if not events.empty:
        msg = 'events DataFrame must have a "type" column with strings'
        assert "type" in events.keys(), msg
        types = events["type"].unique()
        assert all(isinstance(typ, str) for typ in types), msg
        # event-level validation
        df = pd.DataFrame(
            events.apply(_validate_event, axis=1).tolist(),
            index=events.index,
        )
        # add dynamic field
        df = df.assign(stop=lambda x: x.start + x.duration)

        return df
    else:
        return events.copy()


def read_events(events: pd.DataFrame) -> tp.Iterable[tp.Any]:
    """
    Read the data of all events.
    """
    for event in events.itertuples():
        event_type = str(event.type)
        if event_type not in Event._CLASSES:
            yield None

        cls = Event._CLASSES[event_type]
        if not issubclass(cls, BaseDataEvent):
            yield None

        yield cls.from_dict(event._asdict()).read()  # type: ignore


def intersection_segments(
    events: pd.DataFrame,
    starts: float | np.ndarray,
    durations: float | np.ndarray,
    within_only: bool = False,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """Given a start (or array of starts) and duration (or array of durations) providing
    one or more time windows, yield one Segment comprising all events occuring (strictly or not)
    within each time window.
    """
    if events.timeline.nunique() != 1:
        raise RuntimeError("only support a single timeline")
    starts = np.ravel(starts)
    if isinstance(durations, (list, tuple)):
        durations = np.array(durations)
    if not isinstance(durations, np.ndarray):
        durations = durations * np.ones_like(starts)
    stops = np.array(starts + durations)
    starts = starts[:, None]
    stops = stops[:, None]
    estarts = np.array(events.start)[None, :]
    estops = np.array(events.start + events.duration)[None, :]
    # compute selected items for each time window!
    if within_only:
        select = estarts >= starts
        select &= estops <= stops
    else:
        if strict_overlap:
            select = estarts < stops
            select &= estops > starts
        else:
            select = estarts <= stops
            select &= estops >= starts
    for k, (start_, duration) in enumerate(zip(starts, durations)):
        start = float(start_.item())
        yield Segment(events=events.loc[select[k]], start=start, duration=duration)


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    starts = np.arange(
        start, stop - duration + eps, stride
    )  # Ignore windows that don't completely overlap like in `mne.events_from_annotations`
    # with `chunk_duration` other than None.
    durations = np.ones_like(starts) * duration
    return starts, durations


def _iter_segments(
    df: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    within_only: bool = False,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """
    Make an iterator of segments based on specific events (`idx`), a `stride`, or both.

    See `ns.segments.list_segments` for a description of parameters.
    """
    # make sure the selected events match the dataframe
    events = {}
    # create events_list once and for all
    for row in df.itertuples(index=True):
        if row.type in Event._CLASSES:
            events[row.Index] = Event.from_dict(row)

    # Regular division of timeline
    if stride is not None:
        assert isinstance(stride, (int, float))
        assert isinstance(duration, (int, float))
        stride = float(stride)
        duration = float(duration)

    # Specific events
    if idx is not None:

        # ensure index is a pd.Series
        if isinstance(idx, int):
            idx = df.index == idx  # type: ignore

        if not np.any(idx):
            avail = pd.unique(df["type"])
            raise ValueError(
                "Empty trigger events provided to list_segments (first argument)\n"
                f"Available events.type: {avail} (did you forget capitalizing the event name?)"
            )

        # convert value-based index to boolean-based index
        # caution: "idx.dtype is bool" doesnt work anymore when reloaded
        # from parquet, which gets a weird type-like object as type
        if "bool" in str(idx.dtype).lower():  # type: ignore
            idx = df.loc[idx].index  # type: ignore
        # check index
        df.loc[idx]  # pylint: disable=pointless-statement

        assert isinstance(start, (int, float))
        start = float(start)

    triggers: tp.Generator | list | np.ndarray
    for _, tl in df.groupby("timeline", sort=False):
        # If we select the batch based on existing events
        if idx is not None:
            j = tl.index.isin(idx)

            if stride is None:
                starts: tp.Any = tl.loc[j].start + start
                # cant pickle named tuples, and iterrows is too slow:
                triggers = (r._asdict() for r in tl.loc[j].itertuples())  # type: ignore

                # If duration is not specified, use the duration of the selected events
                if duration is None:
                    durations: tp.Any = tl.loc[j].duration
                elif isinstance(duration, tuple):
                    # sample between min and max
                    durations = np.random.uniform(duration[0], duration[1], len(starts))
                else:
                    durations = np.ones_like(starts) * duration

            else:  # Extract sliding windows within each event
                assert start == 0.0
                starts, durations, triggers = [], [], []
                for row in tl.loc[j].itertuples():
                    assert isinstance(row.start, float) and isinstance(row.stop, float)
                    assert isinstance(duration, float)
                    _starts, _durations = _prepare_strided_windows(
                        row.start,
                        row.stop,
                        stride,
                        duration,
                    )
                    starts.append(_starts)
                    durations.append(_durations)
                    triggers.extend([row._asdict()] * len(_starts))  # type: ignore
                starts = np.concatenate(starts)
                durations = np.concatenate(durations)

        # if we select the batch based on a sliding window
        elif stride is not None:
            assert start == 0.0
            assert isinstance(duration, float)
            starts, durations = _prepare_strided_windows(
                tl.start.min(), tl.stop.max(), stride, duration
            )
            triggers = starts

        else:
            # find from a specific time
            assert duration is not None
            assert df.timeline.nunique() == 1
            starts = [start]
            durations = [duration]
            triggers = starts

        # For each segment
        inter_segments = intersection_segments(
            tl,
            starts,
            durations,
            within_only=within_only,
            strict_overlap=strict_overlap,
        )
        # add triggers and events
        for segment, trigger in zip(inter_segments, triggers, strict=True):
            segment._trigger = trigger
            # find corresponding events and preadd them (muuuch faster than online)
            evlist = [events[i] for i in segment.events.index if i in events]
            # quickcheck (make sure there's not an indexing issue)
            if evlist:
                estop = evlist[0].start + evlist[0].duration
                if evlist[0].start > segment.stop or estop < segment.start:
                    raise RuntimeError("Event list attribution to segment failed")
            segment._event_list = evlist
            yield segment


def iter_segments(
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """
    Yield segments based on events and/or a stride.

    See `ns.segments.list_segments` for description of parameters.
    """
    for segment in _iter_segments(
        events,
        idx=idx,
        start=start,
        duration=duration,
        stride=stride,
        within_only=False,
        strict_overlap=strict_overlap,
    ):
        yield segment


def list_segments(
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    strict_overlap: bool = True,
) -> list[Segment]:
    """
    Make a list of segments:
    - based on specific events (a single segment is extracted by event):
        ns.segments.list_segments(df, idx=df.type == "Image")
    - based on sliding windows (entire timeline will be subdivided into potentially
        overlapping segments):
        ns.segments.list_segments(df, stride=1.5, duration=3.)
    - or based on both a list of segments and sliding windows (each event will be subdivided
        into potentially overlapping segments; a window must be fully overlapping with the event
        to be valid):
        df.ns.list_segments(df, idx=df.type == "Image", stride=1.5, duration=3.)

    Parameters
    ----------
    idx: pd.Series
        If provided, list of events to use for defining the segments.
    start: float
        Start time (in seconds) of the segment, with respect to the reference event (or stride).
        E.g. use -1.0 if you want the segment to start 1s before the event.
    duration: optional float
        Duration (in seconds) of the segment (defaults to event duration if only using `idx` to
        extract segments based on specific events).
    stride: optional float
        Stride (in seconds) to use to define sliding window segments.
    """
    return list(
        iter_segments(
            events,
            idx=idx,
            start=start,
            duration=duration,
            stride=stride,
            strict_overlap=strict_overlap,
        )
    )


def find_enclosed(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
) -> pd.Series:
    """Find events that are strictly enclosed within the reference time series.

    Example:
        events:
                |-----A----|
                |--B--|
                |-----C-----|
            |----D----|
        df:
                |-----A----|

    would output the index of event A and B.
    """
    if idx is None:  # find from a specific time
        assert duration is not None
        assert events.timeline.nunique() == 1
        is_enclosed = events.start > start
        is_enclosed &= events.start + events.duration < start + duration
        out = events.index[is_enclosed]
        return pd.Series(out)
    else:
        sel = []
        for segment in _iter_segments(
            events, idx=idx, start=start, duration=duration, stride=None, within_only=True
        ):
            sel.extend(segment.events.index.tolist())
        return pd.Series(sel)


def find_overlap(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | np.ndarray | None = None,
    strict_overlap: bool = True,
) -> pd.Series:
    """Find events that overlap within the reference time series.

    The parameter `strict_overlap` determines whether the overlap duration
    should be strictly positive or not.

    Example:
        events:
                |-----A----|
                |--B--|
                |-----C-----|
            |----D----|
        df:
                |-----A----|

    would output the index of event A, B, C, D
    """
    if idx is None:  # find from a specific time
        assert duration is not None
        assert events.timeline.nunique() == 1
        if strict_overlap:
            has_overlap = (events.start >= start) & (events.start < start + duration)
            has_overlap |= (events.start + events.duration > start) & (
                events.start + events.duration <= start + duration
            )
            has_overlap |= (events.start <= start) & (
                events.start + events.duration >= start + duration
            )
        else:
            has_overlap = (events.start >= start) & (events.start <= start + duration)
            has_overlap |= (events.start + events.duration >= start) & (
                events.start + events.duration <= start + duration
            )
            has_overlap |= (events.start <= start) & (
                events.start + events.duration >= start + duration
            )
        out = events.index[has_overlap]
        return pd.Series(out)
    else:
        sel = []
        for segment in _iter_segments(
            events,
            idx=idx,
            start=start,
            duration=duration,  # type: ignore
            stride=None,
            within_only=False,
            strict_overlap=strict_overlap,
        ):
            sel.extend(segment.events.index.tolist())
        return pd.Series(sel)


def plot_timelines(events: pd.DataFrame, separate_splits: bool = False) -> None:
    """Plot a visual representation of timelines and events."""
    if separate_splits:
        assert "split" in events.columns
        assert set(events.split.dropna().unique()) == {"train", "val", "test"}

    import matplotlib.pyplot as plt

    def plot_line(start, stop, y, label):
        plt.plot(
            [start, stop],
            [y, y],
            color=f"C{j}",
            label=label,
            linewidth=3,
        )

    for _, (timeline, df) in enumerate(events.groupby("timeline")):
        plt.figure(figsize=(10, 2))
        for j, (event_type, tl_events) in enumerate(df.groupby("type")):
            starts = tl_events.start
            stops = tl_events.start + tl_events.duration
            if separate_splits and len(tl_events["split"].unique()) > 1:
                for split, split_events in tl_events.groupby("split"):
                    for k, (start, stop) in enumerate(
                        zip(split_events.start, split_events.stop)
                    ):
                        offset = {"train": -0.3, "val": 0, "test": 0.3}[split]  # type: ignore
                        label = event_type if k == 0 and split == "train" else ""
                        plot_line(start, stop, j + offset, label)
            else:
                for k, (start, stop) in enumerate(zip(starts, stops)):
                    plot_line(start, stop, j, event_type if k == 0 else None)

        plt.title(f"Timeline {timeline}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.yticks([])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.xlabel("Time (s)")
        plt.show()


def remove_invalid_segments(
    segments: tp.Sequence[Segment], event_types: tp.Sequence[tp.Type[Event]]
) -> list[Segment]:
    """
    For each event type, check that all segments contain at least one event of that type.
    If some of them don't, remove them from the list.
    """
    all_invalid_indices = set()
    for event_type in event_types:
        invalid_indices = set()
        subclasses = [
            name for name, cls in Event._CLASSES.items() if issubclass(cls, event_type)
        ]
        for i, segment in enumerate(segments):
            if not any(segment.events.type.isin(subclasses)):
                invalid_indices.add(i)
        if invalid_indices:
            msg = "%s segments did not contain valid events for event type %s and will be removed"
            logger.warning(msg, len(invalid_indices), event_type)
        all_invalid_indices.update(invalid_indices)
    return [segment for i, segment in enumerate(segments) if i not in all_invalid_indices]


@pd.api.extensions.register_dataframe_accessor("ns")
class SegmentAccessor:
    """Accessor for event information stored as a pandas DataFrame.

    Deprecated in favour of functions in `neuralset/segments.py`.

    Alternatively, the definitions of the Event (sub)classes can be inspected in `ns/events.py`.

    For more information about events and the `SegmentAccessor`, see
    `doc/recordings_and_events.md`.  # FIXME
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        msg = "SegmentAccessor (events.ns) is deprecated, use `neuralset.segments.{}()` instead."
        self.deprec_msg = msg

        self._frame = frame
        self._frame = self.validate()

    def read(self) -> tp.Iterable[tp.Any]:
        warnings.warn(self.deprec_msg.format("read_events"), DeprecationWarning)
        return read_events(self._frame)

    def validate(self) -> pd.DataFrame:
        warnings.warn(self.deprec_msg.format("validate_events"), DeprecationWarning)
        return validate_events(self._frame)

    # pylint: disable=unused-argument
    def list(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise RuntimeError(self.deprec_msg.format("list_segments"))

    # pylint: disable=unused-argument
    def iter(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise RuntimeError(self.deprec_msg.format("iter_segments"))

    def list_segments(
        self,
        idx: pd.Series | None = None,
        *,
        start: float = 0.0,
        duration: float | tuple[float, float] | None = None,
        stride: float | None = None,
        strict_overlap: bool = True,
    ) -> tp.List[Segment]:
        warnings.warn(self.deprec_msg.format("list_segments"), DeprecationWarning)
        return list_segments(
            self._frame,
            idx=idx,
            start=start,
            duration=duration,
            stride=stride,
            strict_overlap=strict_overlap,
        )

    def plot_timelines(self, separate_splits: bool = False) -> None:
        warnings.warn(self.deprec_msg.format("plot_timelines"), DeprecationWarning)
        plot_timelines(self._frame, separate_splits=separate_splits)

    def iter_segments(
        self,
        idx: pd.Series | None = None,
        *,
        start: float = 0.0,
        duration: float | tp.Tuple[float, float] | None = None,
        stride: float | None = None,
        strict_overlap: bool = True,
    ) -> tp.Generator[Segment, None, None]:
        warnings.warn(self.deprec_msg.format("iter_segments"), DeprecationWarning)
        yield from iter_segments(
            self._frame,
            idx=idx,
            start=start,
            duration=duration,
            stride=stride,
            strict_overlap=strict_overlap,
        )

    def find_enclosed(
        self,
        idx: int | pd.Series | None = None,
        *,
        start: float = 0.0,
        duration: float | None = None,
    ) -> pd.Series:
        warnings.warn(self.deprec_msg.format("find_enclosed"), DeprecationWarning)
        return find_enclosed(self._frame, idx=idx, start=start, duration=duration)

    def find_overlap(
        self,
        idx: int | pd.Series | None = None,
        *,
        start: float = 0.0,
        duration: float | np.ndarray | None = None,
        strict_overlap: bool = True,
    ) -> pd.Series:
        warnings.warn(self.deprec_msg.format("find_overlap"), DeprecationWarning)
        return find_overlap(
            self._frame,
            idx=idx,
            start=start,
            duration=duration,
            strict_overlap=strict_overlap,
        )
