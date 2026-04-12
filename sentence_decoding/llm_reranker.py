"""
llm_reranker.py  —  NeuroSpeak novel contribution #1
=====================================================
Drop-in LLM re-ranking layer for any EEG word-retrieval pipeline.

Usage (3 lines to add to existing code):
    from llm_reranker import LLMReranker
    reranker = LLMReranker()                            # once at startup
    top_k = reranker.rerank(top_k_words, context_so_far)  # after each retrieval

The reranker takes the model's top-K word candidates and re-orders them
using sentence context via an LLM, without touching the neural model at all.
"""

import os
import re
import json
import time
import anthropic
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "claude-haiku-4-5-20251001"  # fast + cheap for real-time use
FALLBACK_MODEL  = "claude-sonnet-4-6"          # use if haiku is unavailable
MAX_TOKENS      = 256
TEMPERATURE     = 0.0   # deterministic reranking
CACHE_SIZE      = 512   # LRU cache for repeated (context, candidates) pairs


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RerankedResult:
    words:           list[str]          # reranked, best first
    original_words:  list[str]          # original model output order
    scores:          dict[str, float]   # LLM confidence per word (0–1)
    context_used:    str                # sentence context passed to LLM
    rerank_applied:  bool               # False if fell back to original order
    latency_ms:      float              # round-trip time to LLM
    top1:            str = field(init=False)

    def __post_init__(self):
        self.top1 = self.words[0] if self.words else ""

    def __repr__(self):
        pairs = [f"{w}({self.scores.get(w, 0):.2f})" for w in self.words[:5]]
        return f"RerankedResult(top1={self.top1!r}, words={pairs})"


# ── Reranker ──────────────────────────────────────────────────────────────────

class LLMReranker:
    """
    Wraps an Anthropic LLM to rerank EEG word-retrieval candidates
    using accumulated sentence context.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    model : str
        Model to use. Haiku is recommended for latency.
    context_window : int
        Number of previous decoded words to include as context.
    min_candidates : int
        Skip reranking if fewer than this many candidates (fallback to original).
    verbose : bool
        Print debug info per call.
    """

    def __init__(
        self,
        api_key:         Optional[str] = None,
        model:           str  = DEFAULT_MODEL,
        context_window:  int  = 8,
        min_candidates:  int  = 2,
        verbose:         bool = False,
    ):
        self.client         = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model          = model
        self.context_window = context_window
        self.min_candidates = min_candidates
        self.verbose        = verbose
        self._cache: dict[tuple, RerankedResult] = {}
        self._history: list[str] = []   # decoded words so far in current sentence

    # ── Public API ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        candidates: list[str],
        extra_context: str = "",
    ) -> RerankedResult:
        """
        Rerank a list of candidate words using sentence context.

        Parameters
        ----------
        candidates : list[str]
            Top-K words from the neural retrieval step, best-first.
        extra_context : str, optional
            Any extra context string (e.g. known sentence prefix from the
            experimental paradigm). Added before the decoded history.

        Returns
        -------
        RerankedResult
            Always returns something — falls back to original order on error.
        """
        if len(candidates) < self.min_candidates:
            return self._fallback(candidates, "too few candidates")

        context = self._build_context(extra_context)
        cache_key = (context, tuple(candidates))

        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.perf_counter()
        try:
            result = self._call_llm(candidates, context)
        except Exception as e:
            if self.verbose:
                print(f"[LLMReranker] error: {e}")
            result = self._fallback(candidates, str(e))

        result.latency_ms = (time.perf_counter() - t0) * 1000
        if self.verbose:
            print(f"[LLMReranker] {result}  latency={result.latency_ms:.0f}ms")

        if len(self._cache) >= CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result

    def accept(self, word: str):
        """
        Call this after each accepted prediction to update sentence context.
        Skips uncertain / abstained predictions (pass the empty string).
        """
        if word and word != "uncertain":
            self._history.append(word)
            if len(self._history) > self.context_window * 3:
                self._history = self._history[-self.context_window * 3:]

    def reset(self):
        """Call at the start of each new sentence / trial block."""
        self._history.clear()
        self._cache.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_context(self, extra: str) -> str:
        parts = []
        if extra:
            parts.append(extra.strip())
        recent = self._history[-self.context_window:]
        if recent:
            parts.append(" ".join(recent))
        return " ".join(parts).strip()

    def _call_llm(self, candidates: list[str], context: str) -> RerankedResult:
        cand_str = ", ".join(f'"{w}"' for w in candidates)
        context_line = f'Sentence context so far: "{context}"' if context else "No context yet — this is the first word."

        prompt = f"""You are helping decode brain signals into words.
A neural decoder produced these candidate words (in order of neural similarity):
{cand_str}

{context_line}

Task: Rerank these candidates so the most likely next word comes first, given the context.
Consider grammar, semantics, and naturalness.

Respond ONLY with a JSON object, nothing else:
{{
  "ranked": ["word1", "word2", ...],
  "scores": {{"word1": 0.9, "word2": 0.6, ...}}
}}

All original candidates must appear in "ranked". Scores are your confidence (0.0–1.0)."""

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        data = json.loads(raw)
        ranked = data.get("ranked", candidates)
        scores = data.get("scores", {})

        # Safety: keep only words that were in the original candidates
        cand_set = set(candidates)
        ranked_clean = [w for w in ranked if w in cand_set]
        # Add any missing ones at the end (model sometimes drops words)
        for w in candidates:
            if w not in ranked_clean:
                ranked_clean.append(w)

        return RerankedResult(
            words=ranked_clean,
            original_words=candidates,
            scores={w: float(scores.get(w, 0.5)) for w in ranked_clean},
            context_used=context,
            rerank_applied=True,
            latency_ms=0.0,
        )

    def _fallback(self, candidates: list[str], reason: str) -> RerankedResult:
        if self.verbose:
            print(f"[LLMReranker] fallback ({reason})")
        return RerankedResult(
            words=candidates,
            original_words=candidates,
            scores={w: 1.0 / (i + 1) for i, w in enumerate(candidates)},
            context_used="",
            rerank_applied=False,
            latency_ms=0.0,
        )


# ── Sentence-level wrapper ────────────────────────────────────────────────────

class SentenceDecoder:
    """
    Stateful wrapper that accumulates decoded words into sentences
    and feeds context back into the reranker automatically.

    Usage:
        decoder = SentenceDecoder(reranker)
        for eeg_epoch in trial_stream:
            top_k = your_neural_model.retrieve(eeg_epoch)
            result = decoder.step(top_k)
            print(result.top1, result.scores)
        sentence = decoder.flush()
    """

    def __init__(self, reranker: LLMReranker, sentence_boundary: Optional[str] = "."):
        self.reranker  = reranker
        self.boundary  = sentence_boundary
        self._sentence: list[str] = []

    def step(
        self,
        candidates:    list[str],
        extra_context: str = "",
        confidence_threshold: float = 0.0,
    ) -> RerankedResult:
        result = self.reranker.rerank(candidates, extra_context)
        top = result.top1
        top_score = result.scores.get(top, 0.0)

        if top_score >= confidence_threshold and top != "uncertain":
            self._sentence.append(top)
            self.reranker.accept(top)

        if self.boundary and top == self.boundary:
            self.flush()

        return result

    def flush(self) -> str:
        sentence = " ".join(self._sentence)
        self._sentence.clear()
        self.reranker.reset()
        return sentence

    @property
    def current_sentence(self) -> str:
        return " ".join(self._sentence)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing LLMReranker...")
    print("(Needs ANTHROPIC_API_KEY in environment)\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first. Running mock test instead.\n")
        # Mock test without API
        r = LLMReranker.__new__(LLMReranker)
        r._history = ["the", "astronaut", "stared", "at"]
        r.context_window = 8
        r._cache = {}
        context = r._build_context("")
        print(f"Context built from history: '{context}'")
        print("Expected: 'the astronaut stared at'")
        sys.exit(0)

    reranker = LLMReranker(verbose=True)
    decoder  = SentenceDecoder(reranker)

    # Simulate: model decoded "the astronaut stared at" then got confused
    for w in ["the", "astronaut", "stared", "at"]:
        decoder.reranker.accept(w)
        decoder._sentence.append(w)

    # Now decode next word — model's top-5 from neural retrieval
    candidates = ["night", "evening", "week", "hour", "bed"]
    print(f"\nCandidates from neural model: {candidates}")
    print(f"Context: '{decoder.reranker._build_context('')}'")

    result = decoder.step(candidates)

    print(f"\nReranked: {result.words}")
    print(f"Top-1:    {result.top1}")
    print(f"Scores:   {result.scores}")
    print(f"Latency:  {result.latency_ms:.0f} ms")
    print(f"Applied:  {result.rerank_applied}")
