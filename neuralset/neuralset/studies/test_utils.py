# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd


def _make_test_events():
    sentences = [f"This is sentence {i}" for i in range(3)]
    events_list = [
        dict(
            type="Word",
            text=word,
            language="english",
            split="train",
            sequence_id=i,
            start=0,
            duration=1,
            timeline="foo",
        )
        for i, sentence in enumerate(sentences)
        for word in sentence.split(" ")
    ]
    events = pd.DataFrame(events_list)
    return events


def test_add_sentences() -> None:
    from neuralset.studies.utils import add_sentences

    events = _make_test_events()
    events = add_sentences(events)
    assert "Sentence" in events.type.unique()
    assert len(events.query('type=="Sentence"')) == 3
    words = events.query('type=="Word"')
    assert words.sentence.isna().sum() == 0


def test_nastase_concat_array() -> None:
    from .nastase2020 import _ConcatArray

    T = 10
    left = [np.random.rand(4) for _ in range(T)]
    right = [np.random.rand(4) for _ in range(T)]
    carray = _ConcatArray(left, right)
    out = np.array(carray)
    assert out.shape == (8, 10)
