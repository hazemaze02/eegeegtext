# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy

import neuralset as ns

from . import splitting


def create_wav(fp: Path, fs: int = 44100, duration: float = 10) -> None:
    y = np.random.randn(int(duration * fs))
    scipy.io.wavfile.write(fp, fs, y)


def test_deterministic_splitter() -> None:
    with pytest.raises(AssertionError):
        splitter = splitting.DeterministicSplitter(ratios=dict(train=0.5))
    splitter = splitting.DeterministicSplitter(ratios=dict(train=0.5, test=0.5))
    assert splitter("0") == "train"
    assert splitter("1") == "test"
    assert splitter("10101001010101") == "train"


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


def test_set_event_split(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10.1)
    sound = dict(type="Sound", start=0, timeline="foo", filepath=fp)
    words = [
        dict(
            type="Word",
            text="a",
            start=i,
            duration=i + 1,
            language="english",
            timeline="foo",
            split="train" if i % 2 else "test",
        )
        for i in range(11)
    ]
    events_list = [sound] + words
    events = pd.DataFrame(events_list)
    events = ns.segments.validate_events(events)

    # Split the audio
    events2 = splitting.set_event_split(events)
    sounds = events2[events2["type"] == "Sound"]
    assert len(sounds) == 11
    assert all(sounds.offset.values == list(range(11)))

    events3 = splitting.set_event_split(events, min_duration=0.5)
    sounds = events3[events3["type"] == "Sound"]
    assert len(sounds) == 10


# SimilaritySplitter: test for Cluster Assignment
def test_cluster_assignment():
    test_cases = [
        {
            "clusters": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
            "expected_splits": [
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "val",
                "test",
                "test",
                "train",
            ],
            "ratios": {"train": 0.5, "val": 0.3, "test": 0.2},
        },
        {
            "clusters": [0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
            "expected_splits": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "test",
                "val",
            ],
            "ratios": {"train": 0.85, "val": 0.10, "test": 0.05},
        },
        {
            "clusters": [0, 0, 1, 1, 1, 2, 3, 4, 4, 5],
            "expected_splits": [
                "train",
                "train",
                "val",
                "val",
                "val",
                "train",
                "test",
                "test",
                "test",
                "test",
            ],
            "ratios": {"train": 0.3, "val": 0.3, "test": 0.4},
        },
    ]

    for case in test_cases:
        splitter = splitting.SimilaritySplitter(
            feature=ns.features.text.TfidfEmbedding(), ratios=case["ratios"]
        )
        clusters = case["clusters"]
        result = splitter._cluster_assignment(clusters)
        expected_splits = case["expected_splits"]
        assert result == expected_splits


# SimilaritySplitter: test for Sentence Dataset
def test_sentences_dataset():
    mock_events = []
    sentences = [
        "I want to play with the cat",
        "I have a dog",
        "I want to play with the beautiful cat",
        "I am living in Brooklyn",
    ]

    for sentence in sentences:
        mock_events.append(
            {
                "type": "Sentence",
                "start": 0,
                "timeline": "",
                "duration": 0.1,
                "text": sentence,
            }
        )

    # Adding a non-Sentence event
    mock_events.append(
        {
            "type": "Image",
            "start": 0,
            "duration": None,
            "text": "",
        }
    )

    mock_events = pd.DataFrame(mock_events)
    splitter = splitting.SimilaritySplitter(
        feature=ns.features.text.TfidfEmbedding()
    )  # Adjust the path of ns.features
    mock_events = splitter(mock_events)

    # Check all elements that are not of the right type do not have a split value
    not_sentence_event = mock_events[mock_events["type"] != "Sentence"]
    events_empty_split = mock_events[mock_events["split"] == ""]
    assert len(events_empty_split) == len(
        not_sentence_event
    ), "Non-Sentence events should have NaN split values"

    # Check the 2 similar sentences are grouped together
    assert (
        mock_events.loc[0, "split"] == mock_events.loc[2, "split"]
    ), "Similar sentences should be grouped together"

    # Check 2 dissimilar sentences are not grouped together
    assert (
        mock_events.loc[0, "split"] != mock_events.loc[3, "split"]
    ), "Dissimilar sentences should not be grouped together"


# SimilaritySplitter: test for Image Dataset
def create_image(path: Path, pixel_value: int) -> None:
    from PIL import Image

    """
    Generates a synthetic image with a uniform pixel value and saves it to the specified path.
    Args:
        path (Path): Path to save the generated image.
        pixel_value (int): Pixel intensity for the image.
    """
    image_shape = (100, 100)  # Grayscale image with pixel values
    img_array = np.ones(image_shape, dtype=np.uint8) * pixel_value
    img = Image.fromarray(img_array, mode="L")  # 'L' mode for grayscale images
    img.save(path)


@pytest.fixture
def setup_images(tmp_path: Path):
    """
    Fixture to setup images for testing. This fixture creates 4 images with different pixel values
    and returns their paths.
    """
    filepaths = [tmp_path / f"image_{k + 1}.png" for k in range(4)]
    pixel_values = [1, 1, 250, 30]

    for fp, value in zip(filepaths, pixel_values):
        create_image(fp, value)

    return filepaths


def test_images_dataset(setup_images: list[Path]) -> None:
    """
    Tests the image dataset processing by the SimilaritySplitter.
    Args:
        setup_images (list of Path): List of paths to the generated images.
    """
    filepaths = setup_images

    # Initialize a DataFrame directly
    mock_events = pd.DataFrame(
        {
            "type": ["Image"] * len(filepaths) + ["Sentence"],
            "start": [0] * len(filepaths) + [0],
            "timeline": [""] * len(filepaths) + [""] * 1,
            "duration": [0.1] * len(filepaths) + [None],
            "filepath": [str(filepath) for filepath in filepaths]
            + [None],  # Convert Path to str
        }
    )

    # Call the splitter with the DataFrame
    splitter = splitting.SimilaritySplitter(feature=ns.features.image.Image(device="cpu"))
    mock_events = splitter(mock_events)

    # Check all elements that are not of the right type do not have a split value
    not_image_event = mock_events[mock_events["type"] != "Image"]
    events_empty_split = mock_events[mock_events["split"] == ""]

    assert len(events_empty_split) == len(
        not_image_event
    ), "Non-Image events should have NaN split values"

    # Check the 2 similar images are grouped together
    assert (
        mock_events.loc[0, "split"] == mock_events.loc[1, "split"]
    ), "Similar images should be grouped together"

    # Check the 2 dissimilar images are not grouped together
    assert (
        mock_events.loc[0, "split"] != mock_events.loc[2, "split"]
    ), "Dissimilar images should not be grouped together"
