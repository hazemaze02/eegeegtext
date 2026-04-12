# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import subprocess
import typing as tp
from pathlib import Path

import pandas as pd
import pytest
import yaml

from . import base, data
from .segments import list_segments


class A(base._Module):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("req_a",)


class B(A):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("req_b",)


class C(B):
    pass


def test_requirements() -> None:
    assert B.requirements == ("req_a", "req_b")
    assert C.requirements == ("req_a", "req_b")


def test_frequency_yaml() -> None:
    freq = yaml.safe_dump({"data": base.Frequency(10)})
    assert freq == "data: 10.0\n"


def test_data(tmp_path: Path) -> None:
    # List all events of a study as a dataframe
    events = data.StudyLoader(name="TestMeg2023", path=tmp_path).build()

    # Build a list of segments time-locked to specific events
    dset = list_segments(events, idx=events.type == "Image", start=-0.3, duration=0.5)
    assert dset

    # List all events of a study as a dataframe
    events = data.StudyLoader(name="TestFmri2023", path=tmp_path).build()

    # Build a list of segments time-locked to specific events
    dset = list_segments(
        events,
        idx=events.type == "Word",
        start=-0.3,
        duration=0.5,
    )  # noqa
    assert dset


def test_strict_overlap() -> None:
    events = [
        {
            "start": i,
            "duration": 1,
            "stop": i + 1,
            "type": "Word",
            "text": "foo",
            "timeline": "bar",
        }
        for i in range(3)
    ]
    events_df = pd.DataFrame(events)  # type : ignore
    segments = list_segments(
        events_df, idx=events_df.type == "Word", start=0.0, duration=1
    )
    assert [len(s.events) for s in segments] == [1, 1, 1]
    segments = list_segments(
        events_df,
        idx=events_df.type == "Word",
        start=0.0,
        duration=1,
        strict_overlap=False,
    )
    assert [len(s.events) for s in segments] == [2, 3, 2]


@pytest.mark.skipif("IN_GITHUB_ACTION" in os.environ, reason="No header check in CI")
def test_header() -> None:
    lines = Path(__file__).read_text("utf8").splitlines()
    header = "\n".join(itertools.takewhile(lambda l: l.startswith("#"), lines))
    assert len(header.splitlines()) == 5, f"Identified header:\n{header}"
    root = Path(__file__).parents[2]
    assert root.name == "brainai"
    # list of files to check
    tocheck = []
    for sub in ["neuralset", "neuraltrain"]:
        assert root / sub
        output = subprocess.check_output(
            ["find", str(root / sub), "-name", "*.py"], shell=False
        )
        tocheck.extend([Path(p) for p in output.decode().splitlines()])
    # add missing licenses if none already exists
    missing = []
    AUTOADD = True
    for fp in tocheck:
        if "/build/" in str(fp.relative_to(root)):
            continue
        text = Path(fp).read_text("utf8")
        if not text.startswith(header):
            if AUTOADD and not any(x in text.lower() for x in ("license", "copyright")):
                print(f"Automatically adding header to {fp}")
                Path(fp).write_text(header + "\n\n" + text, "utf8")
            missing.append(str(fp))
    if missing:
        missing_str = "\n - ".join(missing)
        raise AssertionError(
            f"Following files are/were missing standard header (see other files):\n - {missing_str}"
        )
