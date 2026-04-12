# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging
import typing as tp
from functools import lru_cache

import numpy as np
import pandas as pd
import pydantic
from tqdm import tqdm

from neuralset.infra.utils import DISCRIMINATOR_FIELD

from . import events as ev
from . import helpers, splitting, utils
from .segments import find_enclosed
from .splitting import set_event_split

logger = logging.getLogger(__name__)


class BaseEnhancer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: str
    # for automatic registration
    _CLASSES: tp.ClassVar[dict[str, type["BaseEnhancer"]]] = {}
    _discriminating_type_adapter: tp.ClassVar[pydantic.TypeAdapter]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        # check params
        super().__pydantic_init_subclass__(**kwargs)
        name = cls.__name__
        if "Base" not in name and not name.startswith("_"):
            if "name" not in cls.model_fields or cls.model_fields["name"].default != name:  # type: ignore
                # unfortunately, this field can't be added dynamically so far :(
                # https://github.com/pydantic/pydantic/issues/1937
                indication = f"name: tp.Literal[{name!r}] = {name!r}"
                raise NotImplementedError(
                    f"Enhancer {name} has incorret/missing name field, add:\n{indication}"
                )
            BaseEnhancer._CLASSES[cls.model_fields["name"].default] = cls  # type: ignore
            # from https://github.com/pydantic/pydantic/issues/7366:
            # The following will create a new type adapter every time a new subclass is created,
            # which is fine if there aren't that many classes (as far as performance goes)
            BaseEnhancer._discriminating_type_adapter = pydantic.TypeAdapter(
                tp.Annotated[
                    tp.Union[tuple(cls._CLASSES.values())],
                    pydantic.Field(discriminator="name"),
                ]
            )

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _parse_into_subclass(
        cls, v: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "BaseEnhancer":
        if cls is BaseEnhancer:
            out = BaseEnhancer._discriminating_type_adapter.validate_python(v)
        else:
            out = handler(v)
        # make sure it is discriminated (keep name field in uid)
        out.__dict__[DISCRIMINATOR_FIELD] = "name"
        return out

    @pydantic.model_serializer
    def _dump(self, info: pydantic.SerializationInfo) -> dict[str, tp.Any]:
        # replace defaults dumper that would ignore non BaseEnhancer fields
        out: dict[str, tp.Any] = {}
        for name, field in self.model_fields.items():
            val = getattr(self, name)
            if not info.exclude_defaults or val != field.default:
                out[name] = val
        return out

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


EnhancerConfig = BaseEnhancer
Enhancer = BaseEnhancer


class AddSentenceToWords(BaseEnhancer):
    """Adds sentence and sentence_char to Word rows in the events dataframe
    based on Text rows

    Parameters
    ----------
    max_unmatched_ratio: float
        maximum ratio of word rows without sentence allowed (raises if above)
    """

    name: tp.Literal["AddSentenceToWords"] = "AddSentenceToWords"
    max_unmatched_ratio: float = 0.0  # raises if did not match enough words
    override_sentences: bool = False
    _exclude_from_cls_uid: tp.Tuple[str, ...] = ("max_unmatched_ratio",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.max_unmatched_ratio < 0 or self.max_unmatched_ratio >= 1:
            raise ValueError("max_unmatched_ratio must be >=0 and <1")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add sentence information to each word event by parsing the
        corresponding full text"""
        if "Sentence" in events.type.unique():
            if not self.override_sentences:
                msg = "Sentence already present in events dataframe"
                raise ValueError(msg)
            events = events[events.type != "Sentence"]
        if "timeline" in events.columns and len(events.timeline.unique()) > 1:
            timelines = []
            desc = "Add sentence to Word based on Text"
            # 1 timeline at a time
            for _, subevents in tqdm(events.groupby("timeline", sort=False), desc=desc):
                timelines.append(self(subevents))
            return pd.concat(timelines, ignore_index=True)
        contexts = events.loc[events.type == "Text"]
        events = events.copy(deep=True)  # avoid border effect
        words = events.query('type == "Word"')
        events.loc[:, "sentence_char"] = np.nan
        events.loc[:, "sentence"] = ""  # reset sentences

        sentences = []
        for context in contexts.itertuples():
            # find words that are enclosed in this context (requires unique timeline)
            sub = events.loc[find_enclosed(events, idx=context.Index)]  # type: ignore
            sel = sub[sub.type == "Word"].index
            if not len(sel):
                raise ValueError("No word overlapping with context")
            wordseq = words.loc[sel].text.tolist()
            lang = getattr(context, "language", None)
            if not isinstance(lang, str):
                raise ValueError(f"Need language for Text field {context}")
            info = pd.DataFrame(_match_text_words(context.text, wordseq, language=lang), index=sel)  # type: ignore
            events.loc[sel, info.columns] = info
            # create sentence events
            context_sentences = [s.to_dict() for s in _extract_sentences(events)]
            subject = getattr(context, "subject", None)
            if subject is not None:
                for s in context_sentences:
                    s["subject"] = subject
            sentences.extend(context_sentences)
        sentence_df = pd.DataFrame(
            [s for s in sentences if s["text"] != MISSING_SENTENCE]
        )
        events = pd.concat([events, sentence_df], ignore_index=True)
        events = events.sort_values("start")
        events = events.reset_index(drop=True)

        words = events.query('type == "Word"')
        ratio = sum(not s or not isinstance(s, str) for s in words.sentence) / len(words)
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            msg = f"Ratio of unmatched words is {ratio:.4f} on {len(words)} words "
            msg += f"while {cls}.{max_unmatched_ratio=}"
            raise RuntimeError(msg)
        return events


MISSING_SENTENCE = "# MISSING SENTENCE #"


def _extract_sentences(events) -> tp.List[ev.Sentence]:
    """Extract sentence events from the words with sentence annotations"""
    words_df = events.loc[events.type == "Word", :]
    sentences = []
    words: tp.List[tp.Any] = []
    eps = 1e-6
    for k, word in enumerate(words_df.itertuples(index=False)):
        if words and words[-1].timeline == word.timeline:
            if word.start < words[-1].start:
                raise ValueError(
                    f"Words are not sorted within a timeline ({words!r} and then {word!r}"
                )
        sentence_end = False
        if k == len(words_df) - 1:  # last word event
            sentence_end = True
            words.append(word)
        if words:
            sentence_end |= words[-1].timeline != word.timeline
            sentence_end |= word.sentence != words[-1].sentence
            sentence_end |= word.sentence_char <= words[-1].sentence_char
            if sentence_end:
                w0 = words[0]
                text = w0.sentence
                if not (isinstance(text, str) and text):
                    text = MISSING_SENTENCE
                sentences.append(
                    ev.Sentence(
                        start=w0.start - eps,
                        duration=words[-1].start
                        + words[-1].duration
                        - w0.start
                        + 2 * eps,
                        timeline=w0.timeline,
                        text=text,
                    )
                )
                words = []
        words.append(word)
    return sentences


class AssignSentenceSplit(BaseEnhancer):
    """Assign a train/val/test split to each sentence in a deterministic fashion,
    so that the same sentence presented to different participants is always in the same split.

    Parameters
    ----------
    min_duration: float
        minimum duration between the start of a split and the start of the next split
    min_words: int
        minimum number of words between the start of a split and the start of next split
    ratios: (float, float, float)
        train/val/test ratios
    seed: int
        seed for the split assignment
    max_unmatched_ratio: float
        maximum ratio of word rows without sentence allowed (raises if above)
    """

    name: tp.Literal["AssignSentenceSplit"] = "AssignSentenceSplit"
    min_duration: float | None = None
    min_words: int | None = None
    ratios: tp.Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    max_unmatched_ratio: float = 0.0  # raises if did not match enough words
    _exclude_from_cls_uid: tp.Tuple[str, ...] = ("max_unmatched_ratio",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not sum(self.ratios) == 1:
            raise ValueError("Split ratios must sum to 1")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if "timeline" not in events.columns:
            events["timeline"] = "#foo#"
        # check unmatched
        words_df = events.loc[events.type == "Word", :]
        ratio = sum(not s or not isinstance(s, str) for s in words_df.sentence) / len(
            words_df
        )
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            raise RuntimeError(
                f"Ratio of words with no sentence match is {ratio:.2f} while {cls}.{max_unmatched_ratio=}"
            )
        sentences = _extract_sentences(events)
        # merge sentences into groups with at least min_duration between groups
        merged = _merge_sentences(
            sentences, min_duration=self.min_duration, min_words=self.min_words
        )
        # split
        ratios = dict(train=self.ratios[0], val=self.ratios[1], test=self.ratios[2])
        ratios = {x: y for x, y in ratios.items() if y > 0}
        if len(ratios) == 1:  # fast track
            events.loc[events.type == "Word", "split"] = list(ratios)[0]
            if tuple(events.timeline.unique()) == ("#foo#",):
                events = events.drop("timeline", axis=1)
            return events
        splitter = splitting.DeterministicSplitter(ratios, seed=self.seed)
        undef = "undefined"
        affectations: tp.Dict[str | float, tp.Tuple[str, ...] | str] = {
            MISSING_SENTENCE: undef
        }
        groups: tp.Dict[str, tp.Set[str]] = {}  # for debugging
        for part in merged:
            string = "".join(s.text for s in part)
            if string not in affectations:
                affectations[string] = splitter(string)
            split = affectations[string]
            for seq in part:
                groups.setdefault(seq.text, set()).add(string)
                if affectations.setdefault(seq.text, split) != split:
                    affectations[seq.text] = undef
                    conflicts = groups[seq.text]
                    logger.warning(
                        'Sequence split "%s" set to undefined because it belongs to conflicting groups: %s',
                        seq.text,
                        conflicts,
                    )
        # set
        valid = ~(np.logical_or(events.sentence.isnull(), events.sentence == ""))
        events.loc[valid, "split"] = (
            events.loc[valid].sentence.apply(str).apply(lambda x: affectations[x])
        )
        events.loc[np.logical_and(~valid, events.type == "Word"), "split"] = undef
        # remove temporary timeline
        if tuple(events.timeline.unique()) == ("#foo#",):
            events = events.drop("timeline", axis=1)
        return events


class AssignKSplits(BaseEnhancer):
    """Assign k splits on each timeline

    Parameters
    ----------
    k: int
        number of splits
    groupby: optional str
        add k split for each of the group (eg: subject/timeline)
    """

    k: int
    groupby: str | None = None
    name: tp.Literal["AssignKSplits"] = "AssignKSplits"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.groupby is not None:
            groups = []
            sub = self.model_validate({"groupby": None, "k": self.k})
            for _, g in events.groupby(self.groupby, sort=False):
                groups.append(sub(g))
            return pd.concat(groups, ignore_index=True)
        if "timeline" in events.columns and len(events.timeline.unique()) > 1:
            timelines = []  # 1 timeline at a time
            groups = [g for _, g in events.groupby("timeline", sort=False)]
            sub_ks = np.ones(len(groups))  # at least 1 per timeline
            if self.k > len(groups):
                to_add = self.k - len(groups)  # additional group to provide
                sub_ks += np.diff(
                    [np.round(to_add * g / len(groups)) for g in range(len(groups) + 1)]
                )
            for sub_k, subevents in zip(sub_ks, groups):
                sub = self.model_validate({"k": int(sub_k)})
                timelines.append(sub(subevents))
            return pd.concat(timelines, ignore_index=True)
        # check unmatched
        tl = events.iloc[0].timeline if "timeline" in events.columns else ""
        events = events.copy(deep=True)
        events.loc[:, "split"] = ""
        sent_df = events.loc[events.type == "Sentence", :]
        if len(sent_df) < self.k:
            raise RuntimeError(
                f"Not enough sentences for {self.k} splits: {sent_df.text}"
            )
        start = min(events.start) - 1
        for n in range(self.k):
            ind = int(np.round((len(sent_df) - 1) * (n + 1) / self.k))
            last = sent_df.iloc[ind]
            stop = last.duration + last.start
            select: tp.Any = events.type == "Word"
            select = np.logical_and(select, events.start >= start)
            select = np.logical_and(select, events.start < stop)
            events.loc[select, "split"] = f"{tl}_split_{n + 1}"
            start = stop
        return events


class AddContextToWords(BaseEnhancer):
    """Add context field of words rows to events dataframe by concatenating
    the sentence fields.

    Parameters
    ----------
    sentence_only: bool
        only use current sentence as context
    max_context_len: None or int
        if not None, caps the context len to a given number of words (counted through whitespaces)
    split_field: str
        field on which to reset contexts. If empty, context is only reset for new timelines.
    """

    name: tp.Literal["AddContextToWords"] = "AddContextToWords"
    sentence_only: bool = True  # only use context from current sentence
    max_context_len: int | None = None  # cut the context after given number of words
    split_field: str = "split"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if hasattr(events, "context"):  # make sure it is typed as str
            events.context = events.context.fillna("").astype(str)
        words = events.loc[events.type == "Word", :]
        past_sentences: tp.List[str] = []
        last_word: tp.Any = None
        contexts = []
        desc = "Add context to words"
        worditer: tp.Iterator[ev.Word] = words.itertuples(index=False)  # type: ignore
        sfield = self.split_field
        if sfield and sfield not in words.columns:
            raise ValueError(f"split_field {sfield!r} is not part of dataframe columns")
        for word in tqdm(worditer, total=len(words), desc=desc, mininterval=10):
            # get rid unmatched words
            sent = word.sentence
            if not (isinstance(sent, str) and sent):
                # possibly restart context if split is not known either
                if sfield and last_word is not None:
                    if getattr(last_word, sfield, "") != getattr(word, sfield, ""):
                        past_sentences = []
                # set no context
                contexts.append("")
                last_word = None
                continue
            if last_word is not None:
                if word.sentence != last_word.sentence:
                    if word.sentence_char <= last_word.sentence_char:
                        if not self.sentence_only:
                            past_sentences.append(last_word.sentence)
                        # if a split field is provided, reset context if splits differ
                        if sfield:
                            if getattr(last_word, sfield, "") != getattr(
                                word, sfield, ""
                            ):
                                past_sentences = []
            # reset context with timeline + check ordering for safety
            if last_word is not None:
                if last_word.timeline != word.timeline:
                    past_sentences = []  # new timeline -> new context
                elif word.start < last_word.start:
                    msg = "Words are not in increasing order "
                    msg += f"({word} follows {last_word})"
                    raise ValueError(msg)
            if word.sentence_char is None or np.isnan(word.sentence_char):
                # word is not correctly match, let's not add a context
                contexts.append("")
                continue
            last_word = word
            last_char = float(word.sentence_char) + len(word.text)
            context = "".join(past_sentences) + word.sentence[: int(last_char)]
            if self.max_context_len is not None:
                context = " ".join(context.split(" ")[-self.max_context_len - 1 :])
            contexts.append(context)
        events.loc[events.type == "Word", "context"] = contexts
        return events


class AssignWordSplitAndContext(BaseEnhancer):
    """Sequence of 3 standard operations:
    - add sentences to words
    - assign train/val/test split
    - add context

    Parameters
    ----------
    min_duration: float
        minimum duration between the start of a split and the start of next split
    min_words: int
        minimum number of words between the start of a split and the start of next split
    ratios: (float, float, float)
        train/val/test ratios
    seed: int
        seed for the split assignment
    max_unmatched_ratio: float
        maximum ratio of word rows without sentence allowed (raises if above)
    sentence_only: bool
        only use current sentence as context
    max_context_len: None or int
        if not None, caps the context len to a given number of words (counted through whitespaces)
    split_field: str
        field on which to reset contexts. If empty, context is only reset for new timelines.
    """

    name: tp.Literal["AssignWordSplitAndContext"] = "AssignWordSplitAndContext"
    min_duration: float | None = None
    min_words: int | None = None
    ratios: tp.Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    max_unmatched_ratio: float = 0.0  # raises if did not match enough words
    sentence_only: bool = True
    max_context_len: int | None = None  # cut the context after given number of words
    override_sentences: bool = False
    split_field: str = "split"
    _exclude_from_cls_uid: tp.Tuple[str, ...] = ("max_unmatched_ratio",)

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        params = [(x, getattr(self, x)) for x in self.model_fields if x != "name"]
        names = ["sentence", "split", "context"]
        enhancers = [self.enhancer(name) for name in names]  # type: ignore
        add_sentence, assign_split, add_context = enhancers
        # check parameters
        expected = {name for e in enhancers for name in e.model_fields if name != "name"}
        diff = expected.symmetric_difference({p[0] for p in params})
        if diff:  # security if API of an enhancer is modified
            raise RuntimeError(
                f"Mismatch between proposed parameters and enhancers parameters: {diff}"
            )
        # apply
        df = add_sentence(events)
        df = assign_split(df)
        df = add_context(df)
        return df

    def enhancer(self, name: tp.Literal["sentence", "split", "context"]) -> BaseEnhancer:
        params = [(x, getattr(self, x)) for x in self.model_fields if x != "name"]
        enhancers: tp.Dict[str, tp.Type[BaseEnhancer]] = {
            "sentence": AddSentenceToWords,
            "split": AssignSentenceSplit,
            "context": AddContextToWords,
        }
        cls = enhancers[name]
        return cls(**{x: y for x, y in params if x in cls.model_fields})  # type: ignore


class AddConcatenationContext(BaseEnhancer):
    """
    Add context to events by concatenating previous events of the same type.
    """

    name: tp.Literal["AddConcatenationContext"] = "AddConcatenationContext"
    event_type: str = "Word"
    sentence_only: bool = False
    max_context_len: int | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.event_type not in ev.Event._CLASSES:
            raise TypeError(f"Event type {self.event_type} not found in events")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        """In place: adds contatenation of previous words to context."""
        words = events.loc[events.type == self.event_type].copy()

        # identify chunks
        previous = words.copy().shift(1)
        split_change = words.split.astype(str) != previous.split.astype(str)
        timeline_change = words.timeline.astype(str) != previous.timeline.astype(str)
        chunk_change = split_change | timeline_change
        if self.sentence_only:
            # chunk is given by sentence change
            sentence_change = words.sequence_id != previous.sequence_id
            chunk_change = chunk_change | sentence_change
        words.loc[words.index, "chunk"] = np.cumsum(chunk_change)

        # add chunk
        for _, df in words.groupby("chunk"):
            if any(df.start.diff() < 0):  # type: ignore
                raise ValueError("Events should be ordered by start time")
            context = df.text
            for i, _ in enumerate(df.itertuples()):
                context_ = context[: i + 1]
                if self.max_context_len is not None:
                    context_ = context_[-self.max_context_len :]
                events.loc[df.index[i], "context"] = " ".join(context_)

        return events


class RemoveMissing(BaseEnhancer):
    """Remove lines of given event type(s) which have a missing field.
    (defaults to removing words without context)
    """

    name: tp.Literal["RemoveMissing"] = "RemoveMissing"
    event_types: str | tp.Sequence[str] = "Word"
    field: str = "context"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        names = helpers.EventTypesHelper(self.event_types).names
        data = events.loc[:, self.field]
        missing = np.logical_or(data.isnull(), data == "")
        return events.loc[np.logical_or(~events.type.isin(names), ~missing)]


class SplitEvents(BaseEnhancer):
    """
    This functions splits long events (e.g. sounds or video) based on the train/val/test assignment of shorter events (e.g. words).
    Note that it operates out-of-place, and returns a new DataFrame with the split events.
    This ensures that there is no leakage when processing the long events with a deep learning model (e.g. Word2Vec).
    It assumes that the splits have already been assigned for the events to use, and that the events to split have a _split method and an offset attribute.
    Example:
        input:
            max_duration: 2
            events:
                sound:    [.......]
                split:    [1112233]
        out:
            events:
                sound1:   [..]
                sound2:     [.]
                sound3:      [..]
                sound4:        [..]
                split:    [1112233]

    """

    name: tp.Literal["SplitEvents"] = "SplitEvents"
    event_type_to_split: str = "Sound"
    event_type_to_use: str = "Word"
    min_duration: float | None = None
    max_duration: float = np.inf

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        for event_type in [self.event_type_to_split, self.event_type_to_use]:
            if event_type not in ev.Event._CLASSES:
                raise ValueError(f"Event type {event_type} not found in events")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        return set_event_split(
            events,
            self.event_type_to_split,
            self.event_type_to_use,
            self.min_duration,
            self.max_duration,
        )


class AddPhonemes(BaseEnhancer):
    name: tp.Literal["AddPhonemes"] = "AddPhonemes"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class AddPartOfSpeech(BaseEnhancer):
    name: tp.Literal["AddPartOfSpeech"] = "AddPartOfSpeech"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# # # # # tooling # # # # #


@lru_cache
def parse_text(text: str, language: str = "") -> tp.Any:
    nlp = utils.get_spacy_model(language=language)
    return nlp(text)


def _merge_sentences(
    sentences: tp.List[ev.Sentence],
    min_duration: float | None = None,
    min_words: int | None = None,
) -> tp.List[tp.List[ev.Sentence]]:
    """Merge consecutive sequences into groups so that there is a span of
    at least min_duration between the start of each group
    """
    out: tp.List[tp.List[ev.Sentence]] = []
    for s in sentences:
        new = True
        if out:
            if min_duration is not None:  # merge if below min_duration
                new &= s.start - out[-1][0].start >= min_duration
            if min_words is not None:
                new &= sum(len(s.text.split()) for s in out[-1]) >= min_words
        if not new:  # change for timelines
            new |= out[-1][-1].timeline != s.timeline  # different timeline
        if new:
            out.append([s])
        else:
            out[-1].append(s)
    return out


def _word_preproc(word: str) -> str:
    """Removes cases and punctuation for matching"""
    # return word.lower().strip('",. ()?!“”')
    return word.lower().strip('",. ()?!\n\t')


def _match_text_words(
    text: str, words: tp.Sequence[str], language: str = ""
) -> tp.List[tp.Dict[str, tp.Any]]:
    """Match words within a text:
    - splits text into words using spacy
    - matches words (without case) at word level
    - matches unmatched words at character level

    Parameters
    ----------
    text: str
        the text into which words need to be looked-up
    words: list of str
        the ordered list of words to find in the text

    Returns
    -------
    list of dict
        a dictionary for each input word, with keys sentence and sentence_char
    """
    # parse document into sentences
    doc = parse_text(text, language=language)
    text_words = [word for sentence in doc.sents for word in sentence]
    text_words_str = [_word_preproc(w.text) for w in text_words]
    text_match, words_match = utils.match_list(
        text_words_str, [_word_preproc(w) for w in words]
    )
    info: tp.List[tp.Dict[str, tp.Any]] = [{"word": word} for word in words]
    mkey = "text_match"
    for tm, wm in zip(text_match, words_match):
        info[wm][mkey] = tm
    # fix missing words with character level matching
    # note: this whole block is for the few missing cases
    todebug = []
    first: tp.Any = None
    last: tp.Any = None
    for k, i in enumerate(info):
        if mkey not in i:
            todebug.append(i)
            if k != len(info) - 1:
                continue
        if mkey in i:
            last = i
        if todebug:
            start = 0
            if first is not None:
                w = text_words[first[mkey]]
                start = w.idx + len(w)
            end = len(text)
            if last is not None:
                w = text_words[last[mkey]]
                end = w.idx
            subtext = text[start:end].lower()
            concat_words = " ".join(_word_preproc(j["word"]) for j in todebug)
            text_match, words_match = utils.match_list(subtext, concat_words)
            word_idx_charnum = [
                (k, c) for k, i in enumerate(todebug) for c in range(len(i["word"]) + 1)
            ]
            for mtext, mwordseq in zip(text_match, words_match):
                idx, charnum = word_idx_charnum[mwordseq]
                todebug[idx].setdefault("votes", []).append(start + mtext - charnum)
            for i in todebug:
                if "votes" not in i:
                    continue  # nothing we can do here
                votes: tp.List[int] = i.pop("votes")
                best_bet = max(votes, key=votes.count)
                count = votes.count(best_bet)
                if count / len(i["word"]) <= 0.5:
                    logger.warning(
                        "Ignoring unreliable matching for '%s' in '%s'",
                        i["word"],
                        subtext,
                    )
                    continue
                # add warning if not perfect match:
                word = i["word"]
                found = text[best_bet : best_bet + len(word)]
                if _word_preproc(word) != _word_preproc(found):
                    logger.warning(
                        "Approximately matched annotated %r with %r in text", word, found
                    )
                # find closest text word/token to figure out sentence
                bounds = [j[mkey] if j is not None else None for j in [first, last]]
                sub = text_words[bounds[0] : bounds[1]]
                ind = bisect.bisect_left(sub, best_bet, key=lambda w: w.idx + len(w))
                i["sentence"] = sub[ind].sent.text_with_ws
                i["sentence_char"] = best_bet - sub[ind].sent[0].idx
            todebug = []  # finished debugging
        if last is not None:
            first = last
            last = None
    # add information
    for i in info:
        i.pop("word")
        if mkey in i:
            tword = text_words[i.pop(mkey)]
            i["sentence_char"] = tword.idx - tword.sent[0].idx
            i["sentence"] = tword.sent.text_with_ws
    # add sentence (but not char) if missing words are surrounded by same sentence
    prev: str | None = None
    missing = []
    for i in info:
        sent = i.get("sentence", None)
        if sent is None:
            missing.append(i)
            continue
        if prev == sent:
            for i2 in missing:
                i2["sentence"] = sent
        missing = []
        prev = sent
    return info
