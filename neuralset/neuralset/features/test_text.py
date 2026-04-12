# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pandas as pd
import pytest
import torch

import neuralset as ns

from ..enhancers import AddConcatenationContext
from . import text


def _make_test_events():
    sentence = 3 * ("This is a sentence for the unit tests").split(" ")
    events_list = [
        dict(
            type="Word",
            text=sentence[i],
            start=i,
            duration=i + 1,
            language="english",
            timeline="foo",
            split="train",
            sequence_id=0,
        )
        for i in range(len(sentence))
    ]
    events = pd.DataFrame(events_list)
    add_context = AddConcatenationContext()
    add_context(events)
    return events


@pytest.mark.parametrize(
    "feature_cls",
    [
        text.WordLength,
        text.WordFrequency,
        text.SpacyEmbedding,
    ],
)
def test_word_embedding(feature_cls: tp.Type[text.BaseText]) -> None:
    events = _make_test_events()
    feature = feature_cls(aggregation="sum")
    feature.prepare(events)
    events_list = feature._events_from_dataframe(events)
    out = feature.get_static(events_list[0])
    assert isinstance(out, torch.Tensor)


@pytest.mark.parametrize("contextualized", [True, False])
@pytest.mark.parametrize("layer", [0, 0.5, 1])
@pytest.mark.parametrize("cache_all_layers", [True, False])
@pytest.mark.parametrize("model_name", ["gpt2", "t5-small"])
def test_llm(
    contextualized: bool, layer: int, cache_all_layers: bool, model_name: str
) -> None:
    events = _make_test_events()
    feature = text.HuggingFaceText(
        aggregation="sum",
        layers=layer,
        contextualized=contextualized,
        cache_all_layers=cache_all_layers,
        model_name=model_name,
    )
    if hasattr(feature, "model_name"):
        assert "xl" not in feature.model_name, "Avoid large models as default"
    feature.prepare(events)
    events_list = feature._events_from_dataframe(events)
    out = feature.get_static(events_list[0])
    assert isinstance(out, torch.Tensor)


def test_tfidf() -> None:
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

    mock_events_df = pd.DataFrame(mock_events)

    feature = text.TfidfEmbedding()
    feature.prepare(mock_events_df)
    events_list = feature._events_from_dataframe(mock_events_df)
    out = feature.get_static(events_list[0])

    assert isinstance(out, torch.Tensor)


def test_llm_explicit_error() -> None:
    feature = text.HuggingFaceText(
        aggregation="sum",
        layers=1,
        contextualized=True,
        cache_all_layers=False,
        model_name="gpt2",
    )
    word = ns.events.Word(text="word", start=0, duration=1, timeline="x")
    with pytest.raises(ValueError):
        _ = feature.get_static(word)


def _make_word() -> ns.events.Word:
    return ns.events.Word(
        text="Hello",
        start=0,
        duration=1,
        language="english",
        timeline="foo",
        context="Hello from Paris!",
    )


def test_llm_long_context() -> None:
    feature = text.HuggingFaceText(
        aggregation="sum", contextualized=True, model_name="gpt2", device="cpu"
    )
    word = _make_word()
    word.context = " ".join([str(k) for k in range(1024)])
    # should work even though context is larger that 1024=maximum (~1500)
    _ = feature(word, 0, 1)


def test_bart() -> None:
    feature = text.HuggingFaceText(
        aggregation="sum",
        contextualized=True,
        model_name="facebook/bart-base",
        device="cpu",
    )
    word = _make_word()
    _ = feature(word, 0, 1)


def test_llm_pretrained() -> None:
    word = _make_word()
    outputs = [
        text.HuggingFaceText(
            aggregation="sum", contextualized=True, device="cpu", pretrained=pretrained
        )(word, 0, 1)
        for pretrained in [True, False]
    ]
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(outputs[0], outputs[1])
