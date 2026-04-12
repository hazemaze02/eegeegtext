# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import urllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import pytest
import requests
import torch

import neuralset as ns
from neuralset.infra import ConfDict


def create_image(
    fp: Path, size: tp.Tuple[int, int] = (128, 128), coeff: float = 255
) -> None:
    fp.parent.mkdir(exist_ok=True, parents=True)
    array = np.random.rand(*size, 3) * coeff
    im = PIL.Image.fromarray(array.astype(np.uint8))
    im.save(fp)


def test_image(tmp_path: Path) -> None:
    # A study is just a dataframe of events
    image_fps = [tmp_path / "images" / f"im{k}.jpg" for k in range(2)]
    for fp in image_fps:
        create_image(fp)

    events = pd.DataFrame([dict(start=10.0, filepath=image_fps[0])])
    events["type"] = "Image"
    events["duration"] = 0.5
    events["timeline"] = "foo"

    # For each event, we need to specify how these discrete events
    # can be converted into a dense time series.
    feature = ns.features.Image(device="cpu")
    data = feature(events, start=10.0, duration=0.5)
    (n_dims,) = data.shape
    assert n_dims > 0
    assert data.max() > 0
    for _ in range(2):
        feature = ns.features.Image(infra=dict(folder=str(tmp_path)), device="cpu")  # type: ignore
        data = feature(events, start=10.0, duration=0.5)
        (n_dims,) = data.shape
        assert data.max() > 0

        # test prepare
        feature.prepare(events)
    # check uids
    uid = "neuralset.features.image.Image._get_latents,1/default"
    assert feature.infra.uid() == uid
    feature_keys = set(ConfDict.from_model(feature, uid=True).keys())
    expected = {
        "name",
        "model_name",
        "event_types",
        "repo",
        "token",
        "duration",
        "frequency",
        "imsize",
        "aggregation",
        "infra",  # provides version
    }
    assert feature_keys == expected
    expected = {"name", "model_name", "repo", "imsize", "infra", "event_types"}
    assert set(feature.infra.config().keys()) == expected


def test_image_infra_override(tmp_path: Path) -> None:
    feature = ns.features.Image(infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    assert feature.infra.gpus_per_node == 1


def make_cat_event(folder: str | Path) -> ns.events.Image:
    fp = Path(folder) / "test-data" / "image.jpg"
    if not fp.exists():
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        fp.parent.mkdir(exist_ok=True)
        warnings.warn("Downloading cat image")
        fp.write_bytes(requests.get(url, stream=True, timeout=10).raw.read())
    return ns.events.Image(start=0, duration=1, filepath=fp, timeline="blublu")


@pytest.fixture
def cat_event(tmp_path: Path) -> tp.Iterator[ns.events.Image]:
    if ns.CACHE_FOLDER.exists():
        tmp_path = ns.CACHE_FOLDER
    yield make_cat_event(tmp_path)


class RecordedOutputs:
    """Creates a callable from another callable with additional features:
    - records the output at each call
    - optionally overrides positional arguments
    """

    def __init__(self, func: tp.Callable[..., tp.Any], **overrides: tp.Any) -> None:
        self._func = func
        self._overrides = overrides
        self.outputs: tp.List[tp.Any] = []

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        kwargs.update(self._overrides)
        self.outputs.append(self._func(*args, **kwargs))
        return self.outputs[-1]

    @classmethod
    def as_mocked_method(cls, method: tp.Any, **overrides: tp.Any):
        record = cls(func=method, **overrides)
        assert method.__self__ is not None, "Not a method (not attached to an object)"  # type: ignore
        setattr(method.__self__, method.__name__, record)  # type: ignore
        return record


@pytest.mark.parametrize("token", ["avg", "cls", "all"])
def test_image_token(
    cat_event: ns.events.Image, token: tp.Literal["avg", "cls", "all"]
) -> None:
    feat = ns.features.Image(device="cpu", token=token)
    out = feat.get_static(cat_event)
    assert out.ndim == 2 if token == "all" else 1


def test_pytorch_vision_model(cat_event: ns.events.Image) -> None:
    feat = ns.features.Image(device="cpu", repo="pytorch/vision", model_name="resnet18")
    try:
        out = feat.get_static(cat_event)
    except urllib.error.HTTPError:
        pytest.skip("torchhub rate limit")
    assert out.shape == (1000,)


def test_openai_clip(cat_event: ns.events.Image) -> None:
    feat = ns.features.Image(
        device="cpu", repo="hf", model_name="openai/clip-vit-base-patch32"
    )
    record = RecordedOutputs.as_mocked_method(
        feat.model._full_predict, text=["a photo of a cat", "a photo of a dog"]
    )
    latent = next(iter(feat._get_latents([cat_event])))
    assert latent.shape == (50, 768)
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    probs = pred.logits_per_image.softmax(dim=1)
    assert 0.99 < probs[0, 0] < 1
    # output
    out = feat.get_static(cat_event)
    assert out.shape == (768,)


@pytest.mark.parametrize("pretrained", (True, False))
def test_openai_clip_layer(cat_event: ns.events.Image, pretrained: bool) -> None:
    feat = ns.features.ImageTransformer(
        device="cpu", model_name="openai/clip-vit-base-patch32", pretrained=pretrained
    )
    record = RecordedOutputs.as_mocked_method(
        feat.model._full_predict, text=["a photo of a cat", "a photo of a dog"]
    )
    latent = next(iter(feat._get_latents([cat_event])))
    assert latent.shape == (13, 2, 768)
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    probs = pred.logits_per_image.softmax(dim=1)
    if pretrained:
        assert 0.99 < probs[0, 0] < 1
    else:
        assert probs[0, 0] < 0.95
    # output
    out = feat.get_static(cat_event)
    assert out.shape == (768,)


def test_hf_dinov2(cat_event: ns.events.Image) -> None:
    feat = ns.features.Image(
        device="cpu",
        repo="hf",
        model_name="facebook/dinov2-small-imagenet1k-1-layer",
    )
    latent = next(iter(feat._get_latents([cat_event])))
    assert latent.shape == (257, 384)
    # empty image
    impath = Path(cat_event.filepath).parent / "empty.png"
    create_image(impath, coeff=0)
    empty_im_event = ns.events.Image(
        filepath=impath, timeline="blublu", start=1, duration=1
    )
    latent2 = next(iter(feat._get_latents([empty_im_event])))
    assert latent2.shape == latent.shape

    # now check labels are correct with the appropriate classif model (hacky)
    feat = ns.features.Image(  # new cache
        device="cpu",
        repo="hf",
        model_name="facebook/dinov2-small-imagenet1k-1-layer",
    )
    from transformers import AutoModelForImageClassification

    feat.model.model = AutoModelForImageClassification.from_pretrained(feat.model_name)
    record = RecordedOutputs.as_mocked_method(feat.model._full_predict)
    try:
        feat._get_latents([cat_event])
    except:  # not the right output layer as we overrode the model
        pass
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    idx = pred.logits.argmax(-1).item()
    assert feat.model.model.config.id2label[idx] == "tabby, tabby cat"  # type: ignore


@pytest.mark.parametrize("imsize", [None, 512])
def test_hog(cat_event: ns.events.Image, imsize: None | int) -> None:
    feat = ns.features.HOG(imsize=imsize)
    features = feat.get_static(cat_event)
    assert len(features) == (149152 if imsize is None else 127008)
    assert all(features >= 0.0)


@pytest.mark.parametrize("imsize", [None, 512])
def test_lbp(cat_event: ns.events.Image, imsize: None | int) -> None:
    feat = ns.features.LBP(imsize=imsize)
    features = feat.get_static(cat_event)
    assert len(features) == 10
    assert all(features >= 0.0)


@pytest.mark.parametrize("imsize", [None, 512])
def test_color_histogram(cat_event: ns.events.Image, imsize: None | int) -> None:
    feat = ns.features.ColorHistogram(imsize=imsize)
    features = feat.get_static(cat_event)
    assert len(features) == 512
    assert all(features >= 0.0)


def _get_rfft2d_output_dimension(
    return_log_psd: bool,
    return_angle: bool,
    average_channels: bool,
    height: int,
    width: int,
) -> int:
    """Get the output of an RFFT2D latent based on the configuration."""
    k = 1 if (return_log_psd ^ return_angle) else 2
    return (
        (1 if average_channels else 3)  # Number of image channels
        * height  # Number of spectral components in first image dim
        * (width // 2 + 1)  # Number of spectral components in second image dim
        * k  # Account for "viewed-as-float" complex numbers
    )


@pytest.mark.parametrize("n_components_to_keep", [None, 10])
@pytest.mark.parametrize("average_channels", [False, True])
@pytest.mark.parametrize("return_log_psd", [False, True])
@pytest.mark.parametrize("return_angle", [False, True])
@pytest.mark.parametrize("imsize", [None, 512])
def test_rfft2d(
    cat_event: ns.events.Image,
    n_components_to_keep: int | None,
    average_channels: bool,
    return_log_psd: bool,
    return_angle: bool,
    imsize: None | int,
) -> None:
    import torchvision.transforms.functional as TF  # noqa

    image = TF.to_tensor(cat_event.read())

    feat = ns.features.RFFT2D(
        n_components_to_keep=n_components_to_keep,
        average_channels=average_channels,
        return_log_psd=return_log_psd,
        return_angle=return_angle,
        imsize=imsize,
    )
    features = feat.get_static(cat_event)
    assert features.ndim == 1
    width = image.shape[1] if n_components_to_keep is None else n_components_to_keep * 2
    height = image.shape[2] if n_components_to_keep is None else n_components_to_keep * 2
    if imsize is None:
        assert len(features) == _get_rfft2d_output_dimension(
            return_log_psd, return_angle, average_channels, width, height
        )

    # Make sure inverse is same as original image
    if (
        n_components_to_keep is None
        and not average_channels
        and not return_log_psd
        and not return_angle
        and imsize is None
    ):
        image2 = feat._ifft(features, average_channels, width, height)
        assert torch.allclose(image, image2, atol=1e-6)
