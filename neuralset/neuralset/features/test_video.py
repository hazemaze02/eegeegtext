# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import neuralset as ns
from neuralset.infra import ConfDict

from .video import _VideoImage, resamp_first_dim

logging.getLogger("neuralset").setLevel(logging.DEBUG)


@pytest.fixture
def video_event(tmp_path: Path) -> tp.Iterator[ns.events.Video]:
    yield make_video_event(tmp_path)


def make_video_event(folder: str | Path) -> ns.events.Video:
    ns.features.Video.install_requirements()
    filepath = Path(folder) / "random_video_6s.mp4"
    filepath.parent.mkdir(exist_ok=True)
    import moviepy as mp

    duration = 6.0
    fps = 4
    width, height = 128, 96
    num_frames = int(duration * fps)
    shape = (num_frames, height, width, 3)
    frames = np.random.randint(0, 256, shape, dtype=np.uint8)

    # Create a MoviePy video clip from the frames
    video_clip = mp.VideoClip(
        lambda t: frames[int(t * fps) % num_frames], duration=duration
    )

    # Write file
    video_clip.write_videofile(str(filepath), fps=fps, codec="libx264", audio=False)
    # make event
    event_dict = dict(type="Video", filepath=filepath, start=0, timeline="foo")
    event = ns.events.Video.from_dict(event_dict)
    return event


def test_resamp_first_dim() -> None:
    data = torch.rand(12, 7, 5)
    assert resamp_first_dim(data, 8).shape == (8, 7, 5)


def test_video_requirements() -> None:
    reqs = ",".join(ns.features.Video.requirements)
    assert "julius" in reqs, "Missing requirement coming from Feature"
    assert "moviepy" in reqs, "Missing requirement coming from Event"


def test_video_image(video_event: ns.events.Video) -> None:
    movie = video_event.read()
    vi = _VideoImage(video=movie, time=12345.12345)
    assert vi.filepath.endswith("random_video_6s.mp4:12345.123")


@pytest.mark.parametrize("decimated", (True, False))
def test_video(video_event: ns.events.Video, tmp_path: Path, decimated: bool) -> None:
    video_event.read()
    im = {"device": "cpu", "name": "ImageTransformer", "infra": {"keep_in_ram": False}}
    video = ns.features.Video(
        frequency=0.5, decimated=decimated, infra={"folder": tmp_path / "cache"}, image=im  # type: ignore
    )
    out = video(video_event, start=0.0, duration=0.5)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768, 1)
    # test out
    df = pd.DataFrame([video_event.to_dict()])
    assert isinstance(df.loc[0, "filepath"], str)


def test_video_image_latent(video_event: ns.events.Video, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    im = {"device": "cpu", "name": "ImageTransformer", "infra": {"keep_in_ram": False}}
    video = ns.features.Video(
        frequency=0.5, decimated=True, infra={"folder": cache}, image=im  # type: ignore
    )
    out = video(video_event, start=0.0, duration=4)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768, 2)
    latent = next(iter(video._get_latents([video_event])))
    assert latent.shape == (13, 2, 768, 4)


@pytest.mark.skipif(
    not Path("/large_experiments/brainai/pretrained").exists(),
    reason="CI cannot access the model path on cluster",
)
def test_optical_flow(video_event: ns.events.Video, tmp_path: Path) -> None:
    pytest.skip("Deactivated for now")
    ns.features.OpticalFlow.install_requirements()
    for cache in (None, tmp_path / "cache"):
        video = ns.features.OpticalFlow(
            frequency=1.0, iters=2, infra={"folder": cache}, device="cpu"  # type: ignore
        )
        out = video(video_event, start=0.0, duration=2.0)
        assert isinstance(out, torch.Tensor)
    # test out
    df = pd.DataFrame([video_event.to_dict()])
    assert isinstance(df.loc[0, "filepath"], str)
    # check uids
    uid = "neuralset.features.video.OpticalFlow,1/frequency=1,iters=2-afb4d93b"
    assert video.infra.uid() == uid
    feature_keys = set(ConfDict.from_model(video, uid=True).keys())
    assert feature_keys == {"iters", "model_name", "frequency"}
