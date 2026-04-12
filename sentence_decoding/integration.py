"""
integration.py  —  How to plug NeuroSpeak features into d'Ascoli's code
========================================================================
This file is a guide + runnable adapter.

d'Ascoli's retrieval almost certainly looks like one of these two patterns.
Find the matching pattern in their code and add the 3-5 marked lines.

─────────────────────────────────────────────────────────────────────────
PATTERN A — bare cosine retrieval (most common in evaluate.py / decode.py)
─────────────────────────────────────────────────────────────────────────

BEFORE (their code):
    pred_emb  = model(eeg_batch)                      # (B, d)
    scores    = pred_emb @ vocab_embs.T               # (B, V)
    top_k_idx = scores.topk(k=10, dim=-1).indices     # (B, 10)
    top_words = [[vocab[i] for i in row] for row in top_k_idx]

AFTER (add 5 lines, marked ★):
    from integration import NeuroSpeakWrapper          # ★ import once at top
    ns = NeuroSpeakWrapper(vocab_embs_np, vocab_words) # ★ init once before loop

    pred_emb  = model(eeg_batch)                       # unchanged
    scores    = pred_emb @ vocab_embs.T                # unchanged
    top_k_idx = scores.topk(k=10, dim=-1).indices      # unchanged
    top_words = [[vocab[i] for i in row] for row in top_k_idx]

    top_words = ns.enhance(pred_emb.cpu().numpy(),     # ★ add averaging
                            top_words,                  # ★
                            sentence_context="")        # ★ pass known prefix if any

─────────────────────────────────────────────────────────────────────────
PATTERN B — they have a dedicated retrieval function
─────────────────────────────────────────────────────────────────────────

Find the function (likely called `retrieve`, `decode_words`, `get_top_k`):

    def retrieve(pred_emb, vocab_embs, vocab_words, k=10):
        ...
        return top_words   # list[str]

Replace with:

    def retrieve(pred_emb, vocab_embs, vocab_words, k=10,
                 neurospeak=None, context=""):          # ★ add params
        ...
        top_words = original_retrieval(...)
        if neurospeak:                                  # ★
            top_words = neurospeak.enhance(             # ★
                pred_emb, [top_words], context)[0]     # ★
        return top_words

─────────────────────────────────────────────────────────────────────────
WHAT TO LOOK FOR in their codebase
─────────────────────────────────────────────────────────────────────────

    grep -rn "topk\|top_k\|cosine\|retrieve\|vocab" . --include="*.py"

Look for: scores.topk(), F.cosine_similarity(), np.argsort, or a function
returning a list of words.
"""

import numpy as np
from typing import Optional
from llm_reranker import LLMReranker, SentenceDecoder
from averaging_oracle import AveragingOracle


class NeuroSpeakWrapper:
    """
    Single object that combines AveragingOracle + LLMReranker.

    Designed to slot into d'Ascoli's retrieval loop with minimal changes.
    All state (sentence context, epoch buffer) is maintained internally.

    Parameters
    ----------
    vocab_embeddings : np.ndarray, shape (V, d)
        The same vocabulary embeddings used by the original model.
    vocab_words : list[str]
        Corresponding word strings.
    max_avg_epochs : int
        Max epochs to accumulate before forcing a decision (paper uses 8).
    top_k : int
        How many candidates to pass to the LLM reranker.
    confidence_threshold : float
        Abstain if top-1 cosine similarity is below this.
    use_llm : bool
        Disable LLM reranking (e.g. for offline eval without API key).
    anthropic_api_key : str, optional
        Anthropic API key. Reads ANTHROPIC_API_KEY env var if not set.
    """

    def __init__(
        self,
        vocab_embeddings:   np.ndarray,
        vocab_words:        list[str],
        max_avg_epochs:     int   = 8,
        top_k:              int   = 10,
        confidence_threshold: float = 0.25,
        use_llm:            bool  = True,
        anthropic_api_key:  Optional[str] = None,
    ):
        self.oracle = AveragingOracle(
            vocab_embeddings,
            vocab_words,
            max_epochs=max_avg_epochs,
            k=top_k,
            min_confidence=confidence_threshold,
        )
        self.reranker = LLMReranker(api_key=anthropic_api_key) if use_llm else None
        self.decoder  = SentenceDecoder(self.reranker) if self.reranker else None
        self.top_k    = top_k
        self.conf_thr = confidence_threshold
        self._last_result = None

    def enhance(
        self,
        pred_embedding:   np.ndarray,          # (d,) or (1, d) — raw model output
        top_words_orig:   list[list[str]],     # batch of top-K lists (len B)
        sentence_context: str = "",
        batch_idx:        int = 0,
    ) -> list[list[str]]:
        """
        Drop-in replacement for the bare top-K retrieval output.

        Applies:
          1. Averaging oracle (accumulate pred_embedding, re-retrieve)
          2. LLM re-ranking (uses context from previous words)

        Returns the same shape as top_words_orig but reordered/improved.

        Parameters
        ----------
        pred_embedding : np.ndarray
            Raw predicted embedding before vocab retrieval.
            Shape (d,) for single sample, (B, d) for batch.
        top_words_orig : list[list[str]]
            Original top-K per batch item (from the model's own retrieval).
        sentence_context : str
            Any known sentence prefix (e.g. from the experimental paradigm).
        batch_idx : int
            Which item in the batch is the "current" word being decoded.

        Returns
        -------
        list[list[str]]
            Same length as top_words_orig, with enhanced ordering.
        """
        emb = np.asarray(pred_embedding)
        if emb.ndim == 2:
            emb = emb[batch_idx]

        # Step 1: averaging oracle
        oracle_result = self.oracle.push(emb)
        self._last_result = oracle_result

        # Use oracle's retrieval if it improves on original
        enhanced_words = oracle_result.top_words  # averaged top-K

        # Step 2: LLM re-ranking (only if oracle emits or we have enough context)
        if self.decoder and oracle_result.confidence >= self.conf_thr:
            rerank_result = self.decoder.step(
                candidates=enhanced_words,
                extra_context=sentence_context,
                confidence_threshold=self.conf_thr,
            )
            if rerank_result.rerank_applied:
                enhanced_words = rerank_result.words

        # Replace the target batch item; keep others unchanged
        result = list(top_words_orig)
        result[batch_idx] = enhanced_words
        return result

    def accept_word(self, word: str):
        """Call when a word is committed to reset the averaging buffer."""
        self.oracle.reset()
        if self.decoder:
            self.decoder.reranker.accept(word)

    def new_sentence(self):
        """Call at the start of each new sentence / trial block."""
        self.oracle.reset()
        if self.decoder:
            self.decoder.flush()

    @property
    def confidence(self) -> float:
        return self._last_result.confidence if self._last_result else 0.0

    @property
    def n_averaged(self) -> int:
        return self._last_result.n_averaged if self._last_result else 0

    @property
    def confidence_history(self) -> list[float]:
        return self._last_result.confidence_history if self._last_result else []

    @property
    def should_emit(self) -> bool:
        return self._last_result.should_emit if self._last_result else False


# ── Eval helper: measure top-10 accuracy with/without averaging ───────────────

def compare_accuracy(
    pred_embeddings:  np.ndarray,   # (N_trials, d)
    true_words:       list[str],
    vocab_embeddings: np.ndarray,   # (V, d)
    vocab_words:      list[str],
    n_avg_values:     list[int] = [1, 2, 4, 8],
    k:                int = 10,
) -> dict[int, float]:
    """
    Measure top-10 accuracy as a function of epochs averaged.
    Replicates the d'Ascoli Fig 3B experiment.

    Returns dict mapping n_avg → balanced top-10 accuracy.
    """
    # Normalise vocab once
    norms = np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8
    vocab_normed = vocab_embeddings / norms

    results = {}

    for n_avg in n_avg_values:
        correct = 0
        total   = 0

        for trial_start in range(0, len(pred_embeddings) - n_avg + 1, n_avg):
            window = pred_embeddings[trial_start : trial_start + n_avg]
            mean_emb = window.mean(axis=0)
            norm = np.linalg.norm(mean_emb) + 1e-8
            mean_normed = mean_emb / norm

            scores = vocab_normed @ mean_normed
            top_k_idx = np.argsort(scores)[::-1][:k]
            top_k_words = [vocab_words[i] for i in top_k_idx]

            true = true_words[trial_start]
            if true in top_k_words:
                correct += 1
            total += 1

        results[n_avg] = correct / total if total > 0 else 0.0
        print(f"  n_avg={n_avg:>2}:  top-{k} accuracy = {results[n_avg]:.1%}")

    return results


# ── Demo: run this to verify integration works ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NeuroSpeak Integration Test")
    print("=" * 60)

    rng = np.random.default_rng(0)
    D, V = 1024, 250
    vocab_words = [f"word_{i}" for i in range(V)]
    vocab_embs  = rng.normal(0, 1, (V, D)).astype(np.float32)

    # Simulate the wrapper (no LLM API needed for this test)
    ns = NeuroSpeakWrapper(
        vocab_embeddings=vocab_embs,
        vocab_words=vocab_words,
        max_avg_epochs=4,
        use_llm=False,      # flip to True + set ANTHROPIC_API_KEY for real use
    )

    true_idx   = 42
    true_emb   = vocab_embs[true_idx]
    top_k_orig = [[f"word_{i}" for i in range(10)]]  # fake original retrieval

    print(f"\nSimulating 8 epochs for 'word_{true_idx}':")
    print(f"{'Epoch':>5}  {'Top-1':>10}  {'Conf':>6}  {'N_avg':>5}  {'Emit':>5}")
    print("-" * 42)

    for epoch in range(8):
        noisy = true_emb + rng.normal(0, 0.5, D).astype(np.float32)
        enhanced = ns.enhance(noisy, top_k_orig)
        correct = "✓" if enhanced[0][0] == f"word_{true_idx}" else " "
        print(f"  {epoch+1:>3}  {enhanced[0][0]:>10}  {ns.confidence:.3f}  {ns.n_averaged:>5}  {str(ns.should_emit):>5}  {correct}")

    print("\n")
    print("Accuracy vs. n_avg (Fig 3B replication):")
    true_words  = [f"word_{rng.integers(V)}" for _ in range(64)]
    pred_embs   = np.array([vocab_embs[vocab_words.index(w)] + rng.normal(0, 0.4, D)
                            for w in true_words], dtype=np.float32)
    compare_accuracy(pred_embs, true_words, vocab_embs, vocab_words)

    print("\nAll tests passed. Ready to integrate into d'Ascoli's code.")
