# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy

import neuralset as ns

from . import audio


def create_wav(fp: Path, fs: int = 44100, duration: float = 10) -> None:
    y = np.random.randn(int(duration * fs))
    scipy.io.wavfile.write(fp, fs, y)


def test_split_audio(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=1, timeline="whatever", filepath=fp)
    split1 = event._split([4])
    assert split1[0].start == 1.0 and split1[0].offset == 0.0
    assert split1[1].start == 5.0 and split1[1].offset == 4.0
    split2 = split1[1]._split([2, 3])
    assert len(split2) == 3
    assert split2[1].start == 7.0 and split2[1].offset == 6.0


def test_mel_spectrum(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    feat = audio.MelSpectrum(frequency=50, device="cpu")
    out = feat(event, start=8, duration=4)
    assert out.shape == (40, 200)


def test_mel_spectrum_no_freq(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    feat = audio.MelSpectrum(device="cpu")
    assert feat.frequency == "native"
    out = feat(event, start=8, duration=4)
    assert feat.frequency == "native"
    assert feat._output_frequency(event) == 125.1
    assert out.shape == (40, 500)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)
def test_wav2vec(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    feat = audio.Wav2Vec(frequency=50, device="cpu")
    out = feat(event, start=8, duration=4)
    assert out.shape == (1024, 200)


@pytest.mark.parametrize("layers", [[1, 2, 3], [-1, 2, -3], [0.3, 0.4], 1, 0.5])
def test_wav2vec_layers(
    tmp_path: Path, layers: int | float | list[int] | list[float]
) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    feat = audio.Wav2Vec(frequency=50, layers=layers, device="cpu")
    out = feat(event, start=8, duration=4)

    assert out.shape == (1024, 200)


def test_wav2vec_cache_all_layers(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    infra = {"folder": tmp_path / "cache"}

    layers = [1, 2, 3]
    feat1 = audio.Wav2Vec(
        frequency=50, layers=layers, cache_all_layers=True, device="cpu", infra=infra  # type: ignore
    )

    layers2 = [3, 4, 5]
    feat2 = audio.Wav2Vec(
        frequency=50, layers=layers2, cache_all_layers=True, device="cpu", infra=infra  # type: ignore
    )
    feat3 = audio.Wav2Vec(
        frequency=50, layers=layers2, cache_all_layers=False, device="cpu", infra=infra  # type: ignore
    )

    out1 = feat1(event, start=8, duration=4)
    out2 = feat2(event, start=8, duration=4)
    out3 = feat3(event, start=8, duration=4)

    assert out1.shape == (1024, 200)
    assert not (out1 == out2).all()
    assert (out2 == out3).all()

    assert feat1.infra.uid_folder() == feat2.infra.uid_folder()
    assert feat1.infra.uid_folder() != feat3.infra.uid_folder()


@pytest.mark.parametrize("layers", [[-1, -2, -1], [-1, 24]])
def test_wav2vec_layers_duplicates(tmp_path: Path, layers: list[int]) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    feat = audio.Wav2Vec(frequency=50, layers=layers, device="cpu")
    with pytest.raises(AssertionError):
        feat(event, start=8, duration=4)


def test_mel_spectrum_size_issue(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=16_000, duration=1.310812)
    event = ns.events.Sound(start=0, timeline="whatever", filepath=fp)
    melspec = audio.MelSpectrum(
        device="cpu",
        n_mels=23,
        n_fft=50,
        hop_length=10,
        in_sampling=1024,
        frequency=1024,
        infra={"folder": tmp_path / "cache"},  # type: ignore
    )
    # this one is a edge case because of last rounding to a sample that does not exist
    out = melspec(event, start=1.310228937325665, duration=0.05000000000001137)
    assert out.shape == (23, 51)
    # a few more tests
    for _ in range(1000):
        seed = np.random.randint(2**32 - 1)
        print(f"Seeding with {seed} for reproducibility")
        rng = np.random.default_rng(seed)
        event.start = rng.uniform()
        melspec(event, start=1.0 + rng.uniform(), duration=rng.uniform())
