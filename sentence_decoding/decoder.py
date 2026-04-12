# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import torch

try:
    import kenlm
except ImportError:
    kenlm = None  # type: ignore
    # kenlm is only needed for optional beam-search decoding.
    # The rest of the pipeline works fine without it.


class BeamState:

    def __init__(
        self, sentence: str = "", score: float = 0.0, lm_state: tp.Any = None
    ):
        self.sentence = sentence
        self.score = score
        self.lm_state = lm_state or (kenlm.State() if kenlm is not None else None)

    def __repr__(self):
        return self.sentence


class Decoder:

    def __init__(
        self,
        lm_path: str,
        beam_size: int = 10,
        max_labels_per_timestep: int = 20,
        lm_weight: float = 1,
        id2word: tp.Dict[int, str] = None,
    ):
        self.beam_size = beam_size
        self.id2word = id2word
        self.max_labels_per_timestep = max_labels_per_timestep
        self.lm_weight = lm_weight
        self.lm = kenlm.Model(os.path.join(lm_path))

    def text_preproc(self, text: str) -> str:
        text = self.tokenizer.do(text)
        return text

    def decode_greedy(self, emissions: torch.Tensor) -> str:

        sentence = ""
        for logits in emissions:
            idx = logits.argmax()
            word = self.id2word[idx.item()]
            sentence += word + " "
        return sentence

    def decode(self, emissions: torch.Tensor) -> str:

        self.beam = [BeamState()]
        self.lm.BeginSentenceWrite(self.beam[0].lm_state)

        for logits in emissions:
            self.step(logits)

        return self.beam[0].sentence

    def step(self, logits: torch.Tensor):

        new_beam = []
        logits = torch.softmax(logits, dim=0)
        idx = logits.argsort(descending=True)
        top_indices = idx[: self.max_labels_per_timestep]

        for hyp in self.beam:
            sentence, score = hyp.sentence, hyp.score
            for idx in top_indices:
                word = self.id2word[idx.item()]
                new_sentence = sentence + " " + word

                new_state = kenlm.State()
                lm_score = self.lm.BaseScore(hyp.lm_state, word, new_state)
                lm_score *= self.lm_weight
                brain_score = torch.log(logits[idx])
                new_score = score + lm_score + brain_score
                new_beam.append(
                    BeamState(sentence=new_sentence, score=new_score, lm_state=new_state)
                )

        new_beam = sorted(new_beam, key=lambda x: x.score, reverse=True)
        self.beam = new_beam[: self.beam_size]
