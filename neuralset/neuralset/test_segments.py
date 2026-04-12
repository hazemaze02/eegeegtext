# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuralset as ns
import neuralset.segments as seg

from . import utils
from .test_splitting import create_wav


def test_segment() -> None:
    data = {
        "type": "Word",
        "start": 0,
        "work": "whatever",
        "timeline": "t",
        "duration": 1,
    }
    df = pd.DataFrame([data, {"stuff": 12}])
    segment = seg.Segment(df, start=0, duration=1)
    assert len(segment.events) == 2
    assert len(segment.event_list) == 1
    assert set(segment.asdict()) == {"events", "start", "duration"}


@pytest.mark.parametrize("reloaded", (True, False))
@pytest.mark.parametrize("validated", (True, False))
def test_find_intersect(reloaded: bool, validated: bool, tmp_path: Path) -> None:
    #  |-----A----|
    #     |--B--|
    #     |-----C-----|
    # |----D----|
    #                |---E---]

    events = pd.DataFrame(
        {
            "type": ["a", "b", "c", "d", "e"],
            "start": [-0.5, 0.0, 0.0, -1.0, 10.0],
            "duration": [2.0, 1.0, 2.0, 2.0, 2.0],
            "timeline": [1, 1, 1, 1, 1],
        }
    )
    if validated:
        with utils.ignore_all():
            events = seg.validate_events(events)

    sel = events.type == "a"
    starts = events.loc[sel].start.to_numpy()
    durations = events.loc[sel].duration.to_numpy()

    # Test case: within_only=True, find enclosed events
    segments = list(
        seg.intersection_segments(events, starts, durations, within_only=True)
    )
    assert len(segments) == 1
    assert "".join(segments[0].events.type) == "ab"

    # Test case: within_only=False, find overlaping events
    segments = list(
        seg.intersection_segments(events, starts, durations, within_only=False)
    )
    assert "".join(segments[0].events.type) == "abcd"

    # Test case: same with isolated event
    sel = events.type == "e"
    starts = events.loc[sel].start.to_numpy()
    durations = events.loc[sel].duration.to_numpy()

    segments = list(
        seg.intersection_segments(events, starts, durations, within_only=False)
    )
    assert "".join(segments[0].events.type) == "e"
    assert isinstance(segments[0].start, float)
    assert isinstance(segments[0].duration, float)

    # test ns api
    events = seg.validate_events(
        pd.DataFrame(
            {
                "type": ["a", "b", "c", "d", "e", "f"],
                "start": [-0.5, 0, 0, -1, 10, 0],
                "duration": [2, 1, 2, 2, 2, 3],
                "timeline": [1, 1, 1, 1, 1, 2],
            }
        )
    )
    if reloaded:
        # dump and reload to test for changed column dtypes
        fp = tmp_path / "data.parquet"
        events.to_parquet(fp)
        events = pd.read_parquet(fp, dtype_backend="numpy_nullable")

    # check _iter index dtype:
    actual = seg.find_overlap(events, events.index[0])
    assert "".join(events.loc[actual].type) == "abcd"

    actual = seg.find_overlap(events, events.type == "a")
    assert "".join(events.loc[actual].type) == "abcd"
    # cannot find overlap from a specific time if multiple timeline
    with pytest.raises(AssertionError):
        actual = seg.find_overlap(events, start=0.0, duration=1.0)
    actual = seg.find_overlap(events.query("timeline==1"), start=0.0, duration=1.0)
    assert "".join(events.loc[actual].type) == "abcd"

    actual = seg.find_enclosed(events, events.type == "a")
    assert "".join(events.loc[actual].type) == "ab"

    actual = seg.find_enclosed(events.query("timeline==1"), start=-0.1, duration=1.2)
    assert "".join(events.loc[actual].type) == "b"

    actual = seg.find_overlap(events, events.type == "f")
    assert "".join(events.loc[actual].type) == "f"

    expected = "ad", "abcd", "abcd", "d", "e", "f"
    actual_segments = seg.list_segments(events, pd.Series(events.index), duration=0.1)
    for act, exp in zip(actual_segments, expected):
        assert act.events.timeline.nunique() == 1
        assert "".join(act.events.type) == exp
    assert isinstance(actual_segments[0]._trigger, dict)

    # test striding window
    dset = seg.list_segments(events, stride=1.5, duration=3.0)
    assert len(dset) == 8


def test_trigger() -> None:
    events = seg.validate_events(
        pd.DataFrame(
            [
                dict(type="Word", start=42.0, duration=1, text="table", timeline="blu"),
                dict(type="Phoneme", start=42.0, duration=1, text="t", timeline="blu"),
            ]
        )
    )
    with pytest.raises(ValueError):
        _ = seg.list_segments(events, events.type == "wrong")
    segs = seg.list_segments(events, events.type == "Word")
    assert len(segs) == 1
    assert len(segs[0]._event_list) == 2, "Event list should be precomputed"  # type: ignore
    assert segs[0]._trigger["type"] == "Word"  # type: ignore
    string = pickle.dumps(segs[0])
    pickle.loads(string)


def test_list_segments_duration() -> None:
    data = [dict(type="Word", start=0, duration=100, text="text", timeline="blu")]
    events = seg.validate_events(pd.DataFrame(data))
    events.ns.validate()
    duration = 3
    segs = seg.list_segments(events, stride=1.234, duration=duration)
    durations = [s.duration for s in segs]
    assert all(d == duration for d in durations), f"Got {durations}"


@pytest.mark.parametrize("stride", [5.0, 10.0, 20.0])
def test_list_segments_idx_stride(stride: float) -> None:
    data = [
        dict(type="Word", start=0, duration=100, text="text", timeline="blu"),
        dict(type="Word", start=200, duration=100, text="text", timeline="blu"),
    ]
    events = pd.DataFrame(data)
    duration = 10
    segs = events.ns.list_segments(
        idx=events.type == "Word",
        duration=duration,
        stride=stride,
    )
    starts = np.array([s.start for s in segs])
    assert (
        ((starts >= 0.0) & (starts < 100.0)) | ((starts >= 200.0) & (starts < 300.0))
    ).all()

    strides = np.diff(starts)
    assert sum(strides == stride) == len(strides) - 1

    durations = [s.duration for s in segs]
    assert all(d == duration for d in durations), f"Got {durations}"


def test_remove_invalid_segments(tmp_path: Path) -> None:
    sentence = ("This is a sentence for the unit tests").split(" ")
    words = [
        dict(
            type="Word",
            text=sentence[i],
            start=i,
            duration=1.0,
            language="english",
            timeline="foo",
            split="train",
        )
        for i in range(len(sentence))
    ]
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    sound = dict(type="Sound", start=3, timeline="foo", filepath=fp)
    events = [sound] + words
    events_df = seg.validate_events(pd.DataFrame(events))

    segments = seg.list_segments(
        events_df,
        events_df.type == "Word",
        start=-0.5,
        duration=1.0,
    )
    filtered_segments = seg.remove_invalid_segments(segments, [ns.events.Sound])
    # the three first words are invalid
    assert len(filtered_segments) == len(segments) - 3


def test_plot_timelines() -> None:
    events = ns.data.StudyLoader(name="MneSample2013", path=ns.CACHE_FOLDER).build()
    seg.plot_timelines(events)
