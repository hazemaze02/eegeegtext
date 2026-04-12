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

import neuralset as ns
from neuralset.infra.cachedict import CacheDict
from neuralset.infra.utils import DISCRIMINATOR_FIELD

from . import enhancers


def _make_test_events() -> tp.Tuple[tp.List[str], pd.DataFrame]:
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
    return sentence, events


def test_context() -> None:
    sentence, events = _make_test_events()
    enhancer = enhancers.AddConcatenationContext()
    events = enhancer(events)
    for i, row in events.iterrows():
        assert isinstance(i, int)
        assert row["context"] == " ".join(sentence[: i + 1])
    # check discriminator field
    assert enhancer.__dict__[DISCRIMINATOR_FIELD] == "name"


def test_enhancers_gwilliams() -> None:
    STUDY_FOLDER = Path("/large_experiments/brainai/shared/studies/")
    # AWS cluster
    if not STUDY_FOLDER.exists():
        STUDY_FOLDER = Path("/storage/datasets01/shared/studies")
    if not STUDY_FOLDER.exists():
        pytest.skip("Skipping as we are not on cluster", allow_module_level=True)
    name = "Armeni2022"
    name = "PallierListen2023"
    name = "PallierRead2023"
    name = "Gwilliams2022"
    config = {
        "name": name,
        "path": STUDY_FOLDER,
        "cache": None,
        "n_timelines": 1,
        "enhancers": [
            {"name": "AddSentenceToWords", "override_sentences": True},
            {"name": "AssignSentenceSplit", "min_duration": 3},
            {"name": "SplitEvents", "max_duration": 10},
            {"name": "AddContextToWords"},  # , "sentence_only": True},
        ],
    }
    events = ns.data.StudyLoader(**config).build()  # type: ignore
    assert "context" in events.columns
    assert "split" in events.columns
    events.loc[
        events.type == "Word", ["text", "sentence_char", "sentence", "split", "context"]
    ].to_csv(f"{name}.csv", sep=";")


def _make_test_dataframe(duplicate: int = -1) -> pd.DataFrame:
    text = (
        "Peut-être bien que cet homme est absurde. Cependant il est moins absurde que le roi, que le vaniteux, "
        "que le businessman et que le buveur. Au moins son travail a-t-il un sens. Quand il allume son réverbère, "
        "c'est comme s'il faisait naître une étoile de plus, ou une fleur. Quand il éteint son réverbère ça endort "
        "la fleur ou l'étoile. C'est une occupation très jolie. C'est véritablement utile puisque c'est joli."
    )
    events = []
    for k, w in enumerate(text.split()):
        events.append(
            {
                "text": w.strip(".,"),
                "type": "Word",
                "start": k,
                "duration": 0.2,
                "timeline": "lpp",
            }
        )
        if k == duplicate:
            events.append(events[-1])
    events.append(
        {
            "type": "Text",
            "text": text,
            "start": -0.1,
            "duration": len(events),
            "timeline": "lpp",
            "language": "french",
            "subject": "test-subject",
        }
    )
    return ns.segments.validate_events(pd.DataFrame(events))


def test_standard_text_enhancement(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    df = _make_test_dataframe()
    df = enhancers.AddSentenceToWords()(df)
    # in a standard pipeline, this could be cached at some point, which may change some values
    cd: tp.Any = CacheDict(folder=tmp_path)
    cd["test"] = df
    df = cd["test"]
    df = cd["test"]
    assert set(df.loc[[37, 38]].type.unique()) == {
        "Word"
    }  # just to be sure we set sentence to Words
    df.loc[[37, 38], "sentence"] = [np.nan, ""]
    df = enhancers.AssignSentenceSplit(max_unmatched_ratio=0.05)(df)
    df = enhancers.AddContextToWords(sentence_only=False)(df)
    assert df.type[2] == "Word"
    assert df.split[2] == "test"  # deterministic split -> should always be test
    assert df.type[70] == "Word"
    assert df.split[70] == "train"  # deterministic split -> should always be train
    assert np.isnan(df.split[0])  # full text -> no split
    last = len(df) - 1
    assert df.context[last].startswith("Quand")  # wshould be a long context
    df = enhancers.AddContextToWords(sentence_only=True)(df)
    assert df.context[last] == "C'est véritablement utile puisque c'est joli"
    df = enhancers.AddContextToWords(max_context_len=3)(df)
    assert df.context[last] == "utile puisque c'est joli"
    # try full pipeline as one enhancer:
    df = _make_test_dataframe()
    df.loc[:, "context"] = np.nan
    df = enhancers.AssignWordSplitAndContext(max_context_len=2)(df)
    assert not recwarn  # setting an item of incompatible dtype (for context column)
    assert df.context[last] == "puisque c'est joli"
    # test sentence
    sentence_row = df.loc[df.type == "Sentence"].iloc[0]
    sentence = ns.events.Event.from_dict(sentence_row)
    assert sentence.type == "Sentence"
    assert sentence.start == pytest.approx(0, abs=1e-4)
    assert sentence.duration == pytest.approx(6.2, abs=1e-4)
    assert sentence_row.subject == "test-subject", "Subject field must be added"


def test_standard_multi_timeline() -> None:
    # try full pipeline as one enhancer:
    df = _make_test_dataframe()
    df2 = df.copy()
    df2.timeline = "tl2"
    df = pd.concat([df, df2], ignore_index=True)
    assert df.type[70] == "Word"
    df.loc[70, "type"] = "Phoneme"
    df = enhancers.AssignWordSplitAndContext(max_context_len=2)(df)
    # override + all the same split (fast track)
    df = enhancers.AssignWordSplitAndContext(
        max_context_len=2, override_sentences=True, ratios=(1, 0, 0)
    )(df)
    assert all(x == "train" for x in df.loc[df.type == "Word"].split)


def test_assign_k_splits() -> None:
    # try full pipeline as one enhancer:
    df = _make_test_dataframe()
    df2 = df.copy()
    df2.timeline = "tl2"
    df = pd.concat([df, df2], ignore_index=True)
    df = enhancers.AssignWordSplitAndContext(max_context_len=2)(df)
    df = enhancers.AssignKSplits(k=3)(df)
    expected = {"lpp_split_1", "tl2_split_1", "tl2_split_2"}
    assert set(df.loc[df.type == "Word"].split) == expected
    df = enhancers.AssignKSplits(k=2, groupby="timeline")(df)
    expected = {"lpp_split_1", "lpp_split_2", "tl2_split_1", "tl2_split_2"}
    assert set(df.loc[df.type == "Word"].split) == expected


def test_merge_sentences() -> None:
    data = [(1, 4, "x"), (5, 3, "x"), (9, 3, "x"), (10, 3, "y"), (20, 3, "y")]
    seqs = [
        ns.events.Sentence(start=s, duration=d, timeline=tl, text="z.")
        for s, d, tl in data
    ]
    merged = enhancers._merge_sentences(seqs, min_duration=6)
    assert tuple(len(ss) for ss in merged) == (2, 1, 1, 1)
    merged = enhancers._merge_sentences(seqs, min_words=3)
    assert tuple(len(ss) for ss in merged) == (3, 2)


@pytest.mark.parametrize(
    "change_timeline,split_field,expected",
    [
        (False, "split", "b b b. c c c. d d d"),
        (True, "split", "d d d"),
        (False, "", "a a a. b b b. c c c. d d d"),
    ],
)
def test_add_context_to_words(
    change_timeline: bool, split_field: str, expected: str
) -> None:
    words = []
    ind = 0
    timeline = "lpp"
    for word in "abcd":
        if word == "d" and change_timeline:
            timeline = "lpp2"
        for i in range(3):
            words.append(
                {
                    "text": word,
                    "type": "Word",
                    "sentence": (f"{word} " * 3).strip() + ". ",
                    "sentence_char": 2 * i,
                    "start": ind,
                    "duration": 0.2,
                    "timeline": timeline,
                    "split": "test" if word == "a" else "train",
                }
            )
            ind += 1
    df = pd.DataFrame(words)
    df = enhancers.AddContextToWords(sentence_only=False, split_field=split_field)(df)
    assert list(df.context)[-1] == expected


def _make_sentence_events(
    text: str = "he runs. she eats",
    words: tp.Sequence[str] = tuple("he runs. she eats".split()),
    add_sentence_to_words: bool = True,
) -> pd.DataFrame:
    """make dataset and apply Sentence enhancer"""
    # context event
    events = [
        dict(
            type="Text",
            start=-0.1,
            duration=len(words),
            text=text,
            timeline="foo",
            language="english",
        )
    ]
    # word events
    events += [
        dict(type="Word", start=float(i), duration=0.1, text=word, timeline="foo")
        for i, word in enumerate(words)
    ]
    df = ns.segments.validate_events(pd.DataFrame(events))
    if add_sentence_to_words:
        df = enhancers.AddSentenceToWords()(df)
    return df


def test_sentence_to_word_standard() -> None:
    # two sentences + punctucation
    text = "he runs. she eats"
    words = "he runs she eats".split()
    df = _make_sentence_events(text, words)
    assert df.iloc[2].sentence == "he runs. "
    assert df.iloc[-1].sentence == "she eats"

    # tokenization
    words = "I don't run".split()
    text = "I don't run"
    df = _make_sentence_events(text, words)
    assert sum(df.type == "Word") == len(words) == 3
    # TODO capitalization
    # TODO paragraphs


def test_sentence_to_word_missing_word(caplog: pytest.LogCaptureFixture) -> None:
    text = "le petit prince"
    words = ("le", "pince")
    caplog.set_level(logging.WARNING)
    df = _make_sentence_events(text, words)
    assert "Approximately matched" in caplog.text
    assert sum(df.type == "Word") == len(words) == 2
    assert df.iloc[2].sentence == df.iloc[3].sentence
    assert (
        df.loc[3].sentence_char == 10
    ), "found 'pince' to start at the wrong character in {text!r}"


@pytest.mark.parametrize(
    "text,words,exp_char",
    [
        ("le petit prince", ("le", "Prince"), 9),
        ("froid d'hiver", ("froid", "hiver"), 8),
        ("froid d'hiver", ("froid", "d'hiver"), 6),
        ("froid d'hiver continental", ("froid", "hiver", "continental"), 14),
        ("froid d'hiver continental", ("froid", "hixer", "continental"), 14),
    ],
)
def test_sentence_to_word_cases(
    text: str, words: tp.Sequence[str], exp_char: int
) -> None:
    df = _make_sentence_events(text, words)
    assert sum(df.type == "Word") == len(words)
    assert df.iloc[0].text == text  # Text
    assert df.iloc[1].text == text  # Sentence
    assert df.iloc[2].sentence == df.iloc[-1].sentence == text
    assert (
        df.iloc[-1].sentence_char == exp_char
    ), "found {words[-1]!r} to start at the wrong character in {text!r}"


@pytest.mark.parametrize(
    "text,words,exp_chars",
    [
        ("froid d'hiver continental", ("froid", "hiver", "continental"), [0, 8, 14]),
        ("froid d'hiver continental", ("froid", "hixer", "continental"), [0, 8, 14]),
        ("froid d'hiver continental", ("froid", "dixer", "continental"), [0, 8, 14]),
        ("froid d'hiver continental", ("froid", "dixer", "continental"), [0, 8, 14]),
        ("aaaa bbbb cccc cccc", ("aaax", "bbbx", "cccc", "cccc"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc cccc", ("aaaa", "bbbx", "cccx", "cccc"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc dddd", ("aaaa", "bbbx", "cccx", "dddd"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc dddd", ("aaaa", "bbbb", "cccx", "dddx"), [0, 5, 10, 15]),
        ("It's a me, Mario", ("It s", "me", "Maro"), [0, 7, 11]),
        ("le petit prince", ("le", "Prince"), [0, 9]),
        (
            '"Salut, ça va ?".\n"Oui, tranquille."',
            ["salut", "va", "oui", "tranquille"],
            [1, 11, 19, 24],
        ),
        (
            '"Salut, ça va ?".\n"Oui, tranquille."',
            ["salut", "va", "oui", "tranxxxxxx"],
            [1, 11, 19, -1],
        ),
        (
            "Salut, ça va ? Oui, tranquille.",
            ["salut", "va", "oui", "tranquille"],
            [0, 10, 0, 5],
        ),
    ],
)
def test_match_text_words(
    text: str, words: tp.Sequence[str], exp_chars: tp.Sequence[int]
) -> None:
    info = enhancers._match_text_words(text, words)
    out_chars = [i.get("sentence_char", -1) for i in info]
    np.testing.assert_array_equal(out_chars, exp_chars)


def test_failure_case() -> None:
    text = "It's supposed to be cement. Well? Well it's copper “Really?”"  # "?" can make a mess for match list
    words = text.split()
    info = enhancers._match_text_words(text, words)
    assert all(i for i in info)


def test_duplicate() -> None:
    text = "It's supposed to be cement."
    words = text.split()
    words = words[:2] + words[1:]
    info = enhancers._match_text_words(text, words)
    assert set(info[1]) == {"sentence"}  # should have sentence but not sentence_char


def test_duplicate_full_pipeline(recwarn: pytest.WarningsRecorder) -> None:
    # try full pipeline as one enhancer:
    df = _make_test_dataframe(duplicate=2)
    df = enhancers.AssignWordSplitAndContext(max_context_len=6)(df)
    assert not recwarn  # setting an item of incompatible dtype (for context column)
    # no restart after duplicate:
    assert df.loc[6].context == "Peut-être bien que cet"


def test_remove_missing():
    df = _make_test_dataframe()
    df.loc[:, "context"] = ""
    df.loc[len(df) // 2 :, "context"] = np.nan
    df.loc[:2, "context"] = "blublu"
    out = enhancers.RemoveMissing()(df)
    assert tuple(out.type) == ("Word", "Word", "Word", "Text")
