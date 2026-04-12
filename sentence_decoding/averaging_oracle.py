"""
averaging_oracle.py  —  NeuroSpeak novel contribution #2
=========================================================
Implements the accumulating-average prediction oracle from d'Ascoli et al. Fig 3B.

The paper shows top-10 accuracy doubles when averaging 8 repetitions of the same word.
This module makes that available in a streaming / real-time setup:

  - Buffer incoming predicted embeddings for the current word slot
  - Re-retrieve top-K from the averaged embedding after each new epoch
  - Track how confidence evolves (for the live demo visualization)
  - Emit a prediction when confidence is stable or max_epochs reached

Usage:
    from averaging_oracle import AveragingOracle
    oracle = AveragingOracle(vocab_embeddings, vocab_words, max_epochs=8)

    for eeg_epoch in stream:
        pred_emb = model(eeg_epoch)            # your model's output
        result   = oracle.push(pred_emb)       # accumulate + retrieve
        print(result.top_words, result.confidence_history)

    oracle.reset()                             # next word slot
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class OracleResult:
    top_words:          list[str]    # best-first after averaging N epochs
    top_scores:         list[float]  # cosine similarity scores
    n_averaged:         int          # how many epochs contributed
    confidence:         float        # max cosine sim of top-1 prediction
    confidence_history: list[float]  # confidence after each epoch (for UI)
    is_stable:          bool         # True if prediction hasn't changed for stability_window epochs
    should_emit:        bool         # True if oracle recommends committing this prediction

    @property
    def top1(self) -> str:
        return self.top_words[0] if self.top_words else ""

    @property
    def top1_score(self) -> float:
        return self.top_scores[0] if self.top_scores else 0.0


# ── Oracle ────────────────────────────────────────────────────────────────────

class AveragingOracle:
    """
    Streaming embedding accumulator for EEG word decoding.

    Accumulates predicted embeddings epoch-by-epoch, re-retrieves top-K
    from the running mean, and signals when the prediction is stable enough
    to commit.

    Parameters
    ----------
    vocab_embeddings : np.ndarray, shape (V, d)
        Pre-computed T5 (or other LM) embeddings for vocabulary words.
    vocab_words : list[str]
        Vocabulary words corresponding to rows of vocab_embeddings.
    max_epochs : int
        Maximum epochs to accumulate before forcing a decision.
    k : int
        Number of top candidates to retrieve (top-K).
    stability_window : int
        Emit "stable" if top-1 word hasn't changed for this many epochs.
    min_confidence : float
        Minimum cosine similarity to not abstain.
    """

    def __init__(
        self,
        vocab_embeddings:  np.ndarray,
        vocab_words:       list[str],
        max_epochs:        int   = 8,
        k:                 int   = 10,
        stability_window:  int   = 3,
        min_confidence:    float = 0.25,
    ):
        assert vocab_embeddings.shape[0] == len(vocab_words), \
            "vocab_embeddings rows must match vocab_words length"

        # L2-normalise vocabulary once at init (faster cosine at query time)
        norms = np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8
        self.vocab_embs   = vocab_embeddings / norms     # (V, d)
        self.vocab_words  = vocab_words
        self.max_epochs   = max_epochs
        self.k            = k
        self.stability_window = stability_window
        self.min_confidence   = min_confidence

        self._buffer: list[np.ndarray] = []   # accumulated predicted embeddings
        self._conf_history: list[float] = []
        self._top1_history: list[str]   = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, pred_embedding: np.ndarray) -> OracleResult:
        """
        Add one predicted embedding and return the current averaged prediction.

        Parameters
        ----------
        pred_embedding : np.ndarray, shape (d,) or (1, d)
            Raw predicted embedding from the neural model (before vocab retrieval).

        Returns
        -------
        OracleResult
        """
        emb = np.asarray(pred_embedding).ravel().astype(np.float32)
        self._buffer.append(emb)

        # Running mean of accumulated embeddings
        mean_emb = np.mean(self._buffer, axis=0)

        # L2-normalise mean before cosine retrieval
        norm = np.linalg.norm(mean_emb) + 1e-8
        mean_emb_normed = mean_emb / norm

        # Cosine similarity against entire vocabulary (fast matmul)
        scores = self.vocab_embs @ mean_emb_normed           # (V,)
        top_idx = np.argsort(scores)[::-1][:self.k]

        top_words  = [self.vocab_words[i] for i in top_idx]
        top_scores = [float(scores[i])    for i in top_idx]
        confidence = top_scores[0] if top_scores else 0.0

        self._conf_history.append(confidence)
        self._top1_history.append(top_words[0] if top_words else "")

        is_stable  = self._check_stability()
        should_emit = (
            is_stable
            or len(self._buffer) >= self.max_epochs
        ) and confidence >= self.min_confidence

        return OracleResult(
            top_words=top_words,
            top_scores=top_scores,
            n_averaged=len(self._buffer),
            confidence=confidence,
            confidence_history=list(self._conf_history),
            is_stable=is_stable,
            should_emit=should_emit,
        )

    def reset(self):
        """Call between word slots to clear the accumulator."""
        self._buffer.clear()
        self._conf_history.clear()
        self._top1_history.clear()

    @property
    def n_accumulated(self) -> int:
        return len(self._buffer)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_stability(self) -> bool:
        if len(self._top1_history) < self.stability_window:
            return False
        recent = self._top1_history[-self.stability_window:]
        return len(set(recent)) == 1   # all same


# ── Batch helper (for offline eval) ───────────────────────────────────────────

def batch_averaged_retrieval(
    pred_embeddings:  np.ndarray,
    vocab_embeddings: np.ndarray,
    vocab_words:      list[str],
    n_avg:            int = 8,
    k:                int = 10,
) -> list[OracleResult]:
    """
    Offline helper: simulate the averaging oracle over a batch of predictions
    for the same word, returning a result after every n_avg epochs.

    Parameters
    ----------
    pred_embeddings : np.ndarray, shape (N, d)
        N consecutive predicted embeddings for the same word stimulus.
    vocab_embeddings : np.ndarray, shape (V, d)
    vocab_words : list[str]
    n_avg : int
        How many epochs to average before returning a result.
    k : int
        Top-K to retrieve.

    Returns
    -------
    list[OracleResult]
        One result per averaging window.
    """
    oracle = AveragingOracle(vocab_embeddings, vocab_words, max_epochs=n_avg, k=k)
    results = []
    for i, emb in enumerate(pred_embeddings):
        result = oracle.push(emb)
        if (i + 1) % n_avg == 0 or result.should_emit:
            results.append(result)
            oracle.reset()
    return results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing AveragingOracle with synthetic embeddings...\n")

    rng  = np.random.default_rng(42)
    D    = 1024   # T5-large hidden dim
    V    = 250    # reduced vocabulary

    # Fake vocabulary
    vocab_words = [f"word_{i}" for i in range(V)]
    vocab_embs  = rng.normal(0, 1, (V, D)).astype(np.float32)

    # Simulate: true word is word_42; model predictions are noisy estimates
    true_emb    = vocab_embs[42]
    noisy_preds = true_emb + rng.normal(0, 0.8, (16, D)).astype(np.float32)

    oracle = AveragingOracle(vocab_embs, vocab_words, max_epochs=8, k=5)

    print(f"{'Epoch':>5}  {'Top-1':>10}  {'Confidence':>11}  {'Stable':>7}  {'Emit':>6}")
    print("-" * 50)
    for i, emb in enumerate(noisy_preds):
        r = oracle.push(emb)
        correct = "✓" if r.top1 == "word_42" else " "
        print(f"  {i+1:>3}  {r.top1:>10}  {r.confidence:>10.4f}  {str(r.is_stable):>7}  {str(r.should_emit):>6}  {correct}")
        if r.should_emit:
            print(f"\n  → Emitting: '{r.top1}' after {r.n_averaged} epochs")
            oracle.reset()
            print()
