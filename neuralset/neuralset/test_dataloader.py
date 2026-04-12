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
from torch.utils.data import DataLoader

import neuralset as ns

from . import dataloader as dl
from .segments import list_segments


def _make_segments() -> list[ns.segments.Segment]:
    events = ns.segments.validate_events(
        pd.DataFrame(
            [
                dict(type="Word", start=10.0, duration=0.5, text="Hello", timeline="x"),
                dict(type="Word", start=12.0, duration=0.5, text="world", timeline="x"),
            ]
        )
    )
    segments = list_segments(events, idx=events.type == "Word", start=0.0, duration=1.0)
    return segments


def test_collate_v2() -> None:
    # A study is just a dataframe of events
    segments: tp.Any = _make_segments()
    pulse = ns.features.Pulse(frequency=100.0, aggregation="sum")
    feats: tp.Any = {"single": pulse}
    collate_fn = dl.CollateSegments(feats)
    dataloader = DataLoader(segments, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["single"].shape == (2, 1, 100)


def test_padded_collate() -> None:
    # A study is just a dataframe of events
    segments: tp.Any = _make_segments()
    segments[1].duration = 2.0
    feats = {"Pulse": ns.features.Pulse(frequency=100.0)}
    collate_fn = dl.CollateSegments(feats)
    dataloader = DataLoader(segments, collate_fn=collate_fn, batch_size=2)
    with pytest.raises(RuntimeError):
        next(iter(dataloader))
    collate_fn = dl.CollateSegments(feats, pad_duration=2.5)
    dataloader = DataLoader(segments, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["Pulse"].shape == (2, 1, 250)


# # # # # dataset option # # # # #


def test_batch_and_collate() -> None:
    batches = []
    for k in range(3):
        segment = ns.segments.Segment(
            start=0,
            duration=1,
            events=pd.DataFrame([{"stuff": 12, "ind": k} for _ in range(2)]),
        )
        b = dl.SegmentData(
            {"feat": 10 * torch.Tensor([[k, k, k, k]])}, segments=[segment]
        )
        batches.append(b)
    assert batches[0].data["feat"].shape == (1, 4)
    assert len(batches[0].segments) == 1
    collate_fn = dl.SegmentDataset({}, []).collate_fn
    batch = collate_fn(batches)
    assert len(batch.segments) == 3
    np.testing.assert_array_equal(batch.data["feat"][:, 0], [0, 10, 20])
    assert batch.data["feat"].shape == (3, 4)
    # sequential
    batch = collate_fn(batches[:2])
    batch = collate_fn([batch, batches[2].to("cpu")])
    np.testing.assert_array_equal(batch.data["feat"][:, 0], [0, 10, 20])
    assert len(batch.segments) == 3


def test_dataset() -> None:
    # A study is just a dataframe of events
    segments = _make_segments()
    pulse = ns.features.Pulse(frequency=100.0)
    feats: tp.Any = {"single": pulse}
    ds = dl.SegmentDataset(feats, segments)
    dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["single"].shape == (2, 1, 100)
    # as one batch
    full_batch = ds.as_one_batch()
    assert full_batch.data["single"].shape == (2, 1, 100)


def test_as_one_batch_order() -> None:
    data = [
        dict(timeline=str(k), start=k, duration=0.1, type="Stimulus", code=k)
        for k in range(130)
    ]
    df = ns.segments.validate_events(pd.DataFrame(data))
    stim = ns.features.Stimulus()
    segments = list_segments(df, idx=df.type == "Stimulus", start=0.0, duration=0.1)
    ds = dl.SegmentDataset({"stim1": stim, "stim2": stim}, segments).as_one_batch(
        num_workers=8
    )
    np.testing.assert_array_equal(ds.data["stim1"], ds.data["stim2"])
    np.testing.assert_array_equal(ds.data["stim1"], np.arange(len(df)))


def test_padded_collate_dataset() -> None:
    # A study is just a dataframe of events
    segments = _make_segments()
    segments[1].duration = 2.0
    feats = {"Pulse": ns.features.Pulse(frequency=100.0)}
    ds = dl.SegmentDataset(feats, segments)
    dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
    with pytest.raises(RuntimeError):
        next(iter(dataloader))
    ds = dl.SegmentDataset(feats, segments, pad_duration=2.0)
    dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["Pulse"].shape == (2, 1, 200)


# # # # # EXAMPLE # # # # #


class ExampleDataLoader(pydantic.BaseModel):
    studies: tp.List[ns.data.StudyLoader]
    features: tp.Mapping[str, ns.features.FeatureConfig]
    event_type: str = "Image"
    start: float = 0.0
    duration: float = 1.0
    batch_size: int = 1
    num_workers: int = 0

    def build(self) -> DataLoader[dl.SegmentData]:
        all_studies = pd.concat([s.build() for s in self.studies])
        for feature in self.features.values():
            feature.install_requirements()
            feature.prepare(all_studies)
        segments = list_segments(
            all_studies,
            idx=all_studies.type == self.event_type,
            start=self.start,
            duration=self.duration,
        )
        ds = dl.SegmentDataset(self.features, segments)
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=ds.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def test_example_dataloader(tmp_path: Path) -> None:
    dloader = ExampleDataLoader(
        studies=[{"name": "TestMeg2023", "path": tmp_path}],  # type: ignore
        features={
            "whatever": {"name": "Pulse", "frequency": 100, "aggregation": "sum"}  # type: ignore
        },
        batch_size=2,
    )
    b = next(iter(dloader.build()))
    assert len(b.segments) == 2
    assert b.data["whatever"].shape == (2, 1, 100)
