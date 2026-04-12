# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import neuralset as ns
from neuralset.infra import ConfDict, TaskInfra

from . import base


class ExternFeat(ns.features.Pulse):
    name: tp.Literal["ExternFeat"] = "ExternFeat"  # type: ignore


ns.features.update_config_feature()


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    features: tp.Sequence[ns.features.FeatureConfig] = ()


def test_features_model() -> None:
    model = Model(features=[{"name": "Pulse"}, {"name": "Meg", "frequency": 12}, {"name": "ExternFeat"}])  # type: ignore
    cfg = ConfDict.from_model(model, uid=True, exclude_defaults=True)
    assert (
        cfg.to_yaml()
        == """features:
- name: Pulse
- frequency: 12.0
  name: Meg
- name: ExternFeat
"""
    )


def test_dynamic_feature() -> None:
    # word pulse
    feature = base.Pulse(frequency=1.0, aggregation="sum")

    event = dict(
        type="Word",
        text="test",
        start=0.0,
        duration=2.0,
        language="english",
        timeline="foo",
    )

    # single event and slicing
    out = feature(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 0, 0]])
    # segment does not overlap with event
    out = feature(event, start=-4.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 0, 0]])
    # overwrite event duration
    feature_ = base.Pulse(frequency=1.0, duration=1.0, aggregation="sum")
    out = feature_(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 0, 0, 0]])
    # event start after segment
    event["start"] = 2.0
    out = feature(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 1, 1]])
    # event start before segment and ends when segment starts
    event["start"] = -2.0
    out = feature(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 0, 0]])
    # event start before segment and ends within segment
    event["duration"] = 4.0
    out = feature(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 0, 0]])
    # segment starts before 0
    out = feature(event, start=-1.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 1, 0]])

    # list of events
    segment = dict(events=pd.DataFrame([event]), start=0, duration=4.0)
    data = feature(**segment)  # type: ignore
    assert data.max() == 1 and data.min() == 0
    assert np.array_equal(data.shape, (1, 4))

    # two sequential events
    kwargs = dict(type="Word", text="test", language="english", timeline="foo")
    events = pd.DataFrame(
        [
            dict(start=0.0, duration=1.0, **kwargs),
            dict(start=2.0, duration=1.0, **kwargs),
        ]
    )
    out = feature(events, start=0.0, duration=4)
    assert np.array_equal(out, [[1, 0, 1, 0]])

    # check two simultaneous events
    events = pd.DataFrame(
        [
            dict(start=1.0, duration=2.0, **kwargs),
            dict(start=2.0, duration=2.0, **kwargs),
        ]
    )
    out = feature(events, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 1, 2, 1]])

    # test static feature
    feature = base.Pulse(aggregation="sum")
    out = feature(events, start=0.0, duration=4.0)
    assert np.array_equal(out, [2.0])

    # TODO wordpulse + phonemepulse?


def test_stimulus() -> None:
    code = 1
    feature = base.Stimulus()
    event = dict(
        type="Stimulus",
        code=code,
        description="test",
        start=0.0,
        duration=2.0,
        timeline="foo",
    )
    out = feature(event, start=0.0, duration=4.0)
    assert out == torch.tensor(code)


def test_get_slice() -> None:
    dynamic_feature = base.Pulse(frequency=100.0)
    freq = 100.0
    start = 10.0
    duration = 5.0
    decim = 1

    # Test case: duration >= 0
    expected_slice = slice(1000, 1500, 1)
    assert dynamic_feature._get_slice(freq, start, duration, decim) == expected_slice

    # Test case: duration < 0 (should raise an assertion error)
    with pytest.raises(AssertionError):
        dynamic_feature._get_slice(freq, start, -5.0, decim)


def test_get_overlap_slice() -> None:  #
    freq = 50.0
    segment_start = 8.0
    segment_duration = 7.0

    dynamic_feature = base.Pulse(frequency=freq)
    event_start = 10.0
    event_duration = 6.0

    # Test case: overlap exists
    expected_out_slice = slice(100, 350, 1)
    expected_event_slice = slice(0, 250, 1)
    assert dynamic_feature._get_overlap_slice(
        freq,
        event_start,
        event_duration,
        segment_start,
        segment_duration,
    ) == (expected_out_slice, expected_event_slice)

    # Test case: no overlap
    out_slice, event_slice = dynamic_feature._get_overlap_slice(
        freq, event_start, event_duration, 1000.0, 3.0
    )
    assert out_slice == slice(0, 0, 1)
    assert event_slice == slice(99_000 // 2, 99_000 // 2, 1)

    # Test case: borderline
    for k, start in enumerate([0.05, 0.050000000000000044]):
        out_slice, event_slice = dynamic_feature._get_overlap_slice(
            freq=50,
            event_start=0,
            event_duration=10,
            segment_start=-start,
            segment_duration=3,
        )
        assert out_slice == slice(2 + k, 150, 1)
        assert event_slice == slice(0, 148 - k, 1)


def test_get_overlap() -> None:
    dynamic_feature = base.Pulse(frequency=10.0)
    event_start = 10.0
    event_duration = 6.0
    segment_start = 8.0
    segment_duration = 7.0

    # Test case: overlap exists
    expected_overlap_start = 10.0
    expected_overlap_duration = 5.0
    assert dynamic_feature._get_overlap(
        event_start, event_duration, segment_start, segment_duration
    ) == (expected_overlap_start, expected_overlap_duration)

    # Test case: no overlap
    assert dynamic_feature._get_overlap(event_start, event_duration, 20.0, 3.0) == (
        20.0,
        0.0,
    )


class Time(base.BaseDynamic):
    """Simple dynamic feature for testing fill_slice"""

    name: tp.Literal["Time"] = "Time"
    event_types: str | tp.Tuple[str, ...] = "BaseDataEvent"
    frequency: float = 50.0  # default output frequency

    def _get(
        self, event: ns.events.BaseDataEvent, start: float, duration: float
    ) -> torch.Tensor:
        num = max(1, base.Frequency(self.frequency).to_ind(event.duration))
        latents = np.linspace(event.start, event.start + event.duration, num)[None, :]
        return self._fill_slice(latents, event, start, duration)


@pytest.mark.parametrize(
    "start,duration,expected",
    [
        (0.5, 0.6, [0.6, 0.8, 1, 0]),  # side 1
        (-0.5, 1, [0, 0, 0, 0, 0.2, 0.4]),  # side 2
        (0.2, 0.6, [0.2, 0.4, 0.6, 0.8]),  # inside
        (-0.2, 1.4, [0, 0, 0.2, 0.4, 0.6, 0.8, 1, 0]),  # around
        (0.5, 0.1, [0.6]),  # small (half freq)
        (0.5, 0.01, [0.6]),  # smaller  (we may want to decide there is no sample?)
        (2, 1, [0, 0, 0, 0, 0, 0]),  # no overlap after
        (-2, 1, [0, 0, 0, 0, 0, 0]),  # no overlap before
    ],
)
def test_fill_slice(start: float, duration: float, expected: tp.List[float]) -> None:
    feat = Time(frequency=6, event_types=("Image", "Text"))
    event = ns.events.Image(start=0, duration=1, timeline="stuff", filepath=__file__)
    out = feat(event, start=start, duration=duration)[0]
    np.testing.assert_array_almost_equal(out, expected)


def test_fill_small_event() -> None:
    feat = Time(frequency=6)
    event = ns.events.Image(start=0.5, duration=0.01, timeline="stuff", filepath=__file__)
    out = feat(event, start=0, duration=1)[0]
    np.testing.assert_array_almost_equal(out, [0, 0, 0, 0.5, 0, 0])


@pytest.mark.parametrize(
    "aggreg,expected",
    [
        ("first", [0, 0.5, 1.0]),
        ("trigger", [0, 0, 0.5]),
        ("sum", [0, 0.5, 1.0]),
        ("average", [0, 0.5, 1.0]),
    ],
)
def test_trigger(
    aggreg: tp.Literal["first", "trigger"], expected: tp.List[float]
) -> None:
    feat = Time(frequency=3, aggregation=aggreg)
    event = ns.events.Image(start=0, duration=1, timeline="stuff", filepath=__file__)
    trigger = ns.events.Image(
        start=0.5, duration=1, timeline="stuff", filepath=__file__
    ).to_dict()
    out = feat(event, start=0, duration=1, trigger=trigger)[0]
    np.testing.assert_array_almost_equal(out, expected)
    # exceptions
    if aggreg == "trigger":
        with pytest.raises(ValueError):
            feat(event, start=0, duration=1, trigger=None)


def test_fill_slice_load_testing() -> None:
    seed = np.random.randint(2**32 - 1)
    for k in range(1000):
        print(f"Seeding with {seed + k} for reproducibility")
        rng = np.random.default_rng(seed + k)
        freq = rng.uniform(0, 50)
        if freq > 1 and rng.integers(2):
            freq = int(freq)
        feat = Time(frequency=freq)
        event = ns.events.Image(
            start=rng.uniform(0, 1),
            duration=rng.uniform(0, 1),
            timeline="stuff",
            filepath=__file__,
        )
        start = event.start + rng.uniform(-0.1, 0.1)
        end = event.start + event.duration + rng.uniform(-0.1, 0.1)
        duration = end - start
        if duration < 0:
            duration = rng.uniform(0.1, 0.2)
        feat(event, start=start, duration=duration)


def test_meg_border_cases(tmp_path: Path) -> None:
    # neuro has its own fill slice to deal with mne, so
    # we need a similar check
    loader = ns.data.StudyLoader(
        name="TestMeg2023", path=tmp_path / "data", n_timelines=1
    )
    events = loader.build()
    event = ns.events.Meg.from_dict(events.query('type=="Meg"').iloc[0])
    seed = np.random.randint(2**32 - 1001)
    for k in range(20):
        print(f"Seeding with {seed + k} for reproducibility")
        rng = np.random.default_rng(seed + k)
        feature = ns.features.Meg(frequency=rng.uniform(10, 50))
        start = event.start + rng.uniform(-1, 2) * event.duration
        duration = event.duration * rng.uniform(0, 1)
        out = feature(event, start=start, duration=duration)
        time_inds = max(1, base.Frequency(feature.frequency).to_ind(duration))
        assert out.shape[1] == time_inds


class Xp(pydantic.BaseModel):
    param1: int = 12
    feature: ns.features.FeatureConfig = ns.features.Pulse()
    infra: TaskInfra = TaskInfra()

    @infra.apply
    def run(self) -> int:
        return self.param1


def test_cfg_feature_uid(tmp_path: Path) -> None:
    xp = Xp(feature={"name": "Image", "infra": {"folder": tmp_path}})  # type: ignore
    cfg = xp.infra.config(uid=True, exclude_defaults=True)
    assert cfg["feature.name"] == "Image"


def test_label_encoder(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="TestMeg2023", path=tmp_path / "data", n_timelines=2
    ).build()

    meg_feature = base.LabelEncoder(
        event_types="Meg",
        event_field="filepath",
        return_one_hot=False,
    )
    img_feature = base.LabelEncoder(
        event_types="Image",
        event_field="filepath",
        return_one_hot=False,
    )
    meg_feature.prepare(events)
    img_feature.prepare(events)

    features = {"Meg": meg_feature, "Image": img_feature}
    segments = ns.segments.list_segments(
        events,
        idx=events.type == "Image",
        start=0.0,
        duration=1.0,
    )
    dataset = ns.SegmentDataset(
        features,
        segments,
    )
    out_inds = dataset.as_one_batch().data

    assert (out_inds["Meg"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])).all()
    assert (out_inds["Image"] == torch.tensor([2, 3, 1, 0, 2, 3, 1, 0])).all()


@pytest.mark.parametrize("return_one_hot", [False, True])
def test_label_encoder_one_hot(tmp_path: Path, return_one_hot: bool) -> None:
    events = ns.data.StudyLoader(
        name="TestMeg2023", path=tmp_path / "data", n_timelines=2
    ).build()

    event_field = "filepath"
    feature = base.LabelEncoder(
        event_types="Image",
        event_field=event_field,
        return_one_hot=return_one_hot,
    )
    feature.prepare(events)
    img_events = events[events.type == "Image"]

    inds = []
    for _, ev in img_events.iterrows():
        event = ns.events.Event.from_dict(ev.to_dict())
        inds.append(feature(event, start=0, duration=1))

    out_inds = torch.stack(inds, dim=0)
    assert len(out_inds) == img_events.shape[0]

    gt_event_field = img_events[event_field].to_numpy().reshape(-1, 1)
    if return_one_hot:
        gt_inds = OneHotEncoder(dtype=int, sparse_output=False).fit_transform(
            gt_event_field
        )
    else:
        gt_inds = OrdinalEncoder(dtype=int).fit_transform(gt_event_field)[:, 0]
    assert (out_inds.numpy() == gt_inds).all()
