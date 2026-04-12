# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import mne
import numpy as np
import PIL.Image
import pytest
import scipy.io.wavfile

import neuralset as ns


def test_lru(tmp_path: Path) -> None:
    fp = tmp_path / ".testdata/test.png"
    if not fp.exists():
        fp.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.fromarray(255 * np.ones((100, 100, 3), dtype=np.uint8))
        img.save(str(fp))
    event = ns.events.Image(filepath=fp, start=0, duration=1, timeline="foo")
    event.read()
    # repeat for lru cache
    event.read()


def test_event_from_dict_with_nan() -> None:
    data = {
        "type": "Word",
        "start": 0,
        "timeline": "t",
        "word": "hello",
        "language": np.nan,
        "duration": np.nan,
        "sentence_char": np.nan,
    }
    ns.events.Event.from_dict(data)


def test_sound(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    Fs = 120
    y = np.random.randn(2 * Fs)
    scipy.io.wavfile.write(fp, Fs, y)
    event = ns.events.Event.from_dict(
        dict(type="Sound", start=1, timeline="whatever", filepath=fp, duration=np.nan)
    )
    assert event.duration == 2.0


def test_meg(tmp_path: Path) -> None:
    fp = tmp_path / "test_raw.fif"
    n_channels, sfreq, duration = 4, 64, 60
    data = np.random.rand(n_channels, sfreq * duration)
    info = mne.create_info(n_channels, sfreq=sfreq)
    raw = mne.io.RawArray(data, info=info)
    raw.save(fp)

    event = ns.events.Event.from_dict(
        dict(
            type="Meg",
            start=0,
            timeline="whatever",
            subject="1",
            filepath=str(fp),
        )
    )
    assert isinstance(event, ns.events.Meg)
    assert event.frequency == sfreq
    assert pytest.approx(event.duration, 1 / sfreq) == duration

    raw2 = event.read()
    assert isinstance(raw2, mne.io.Raw)


def test_stimulus() -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Stimulus",
            start=1,
            duration=2,
            timeline="whatever",
            code=1,
            description="right_nostril_smell",
        )
    )
    assert isinstance(event, ns.events.Stimulus)


@pytest.mark.parametrize("state", ("open", "closed"))
def test_eye_state(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="EyeState",
            start=1,
            duration=2,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, ns.events.EyeState)
    assert event.state == state


@pytest.mark.parametrize(
    "state",
    (
        "bckg",
        "seiz",
        "gnsz",
        "fnsz",
        "spsz",
        "cpsz",
        "absz",
        "tnsz",
        "cnsz",
        "tcsz",
        "atsz",
        "mysz",
    ),
)
def test_seizure(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Seizure",
            start=0,
            duration=10,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, ns.events.Seizure)
    assert event.state == state


@pytest.mark.parametrize("state", ("eyem", "musc", "chew", "shiv", "elpp", "artf"))
def test_artifact(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Artifact",
            start=1,
            duration=2,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, ns.events.Artifact)
    assert event.state == state


@pytest.mark.parametrize("key", ("a", "<space>"))
def test_button(key: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Button",
            start=1,
            duration=2,
            timeline="whatever",
            text=key,
        )
    )
    assert isinstance(event, ns.events.Button)
    assert event.text == key


@pytest.mark.parametrize("state", ("spsw", "gped", "pled", "bckg"))
def test_epileptiform_activity(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="EpileptiformActivity",
            start=0,
            duration=1,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, ns.events.EpileptiformActivity)
    assert event.state == state


@pytest.mark.parametrize("stage", ("W", "N1", "N2", "N3", "R"))
def test_sleepstage(stage: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="SleepStage",
            start=0,
            duration=1,
            timeline="whatever",
            stage=stage,
        )
    )
    assert isinstance(event, ns.events.SleepStage)
    assert event.stage == stage
