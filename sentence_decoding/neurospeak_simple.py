"""
neurospeak_simple.py
====================
End-to-end EEG word decoding + LLM sentence reconstruction.
Uses the Broderick (2018) public EEG dataset.

Pipeline:
  1. Load + preprocess EEG  (MNE)
  2. Epoch around word onsets
  3. Predict T5 word embeddings  (ridge regression — no GPU needed)
  4. Retrieve top-K candidates per word  (cosine similarity)
  5. LLM reconstructs most coherent sentence  (Claude API)

Install:
  pip install mne scikit-learn transformers torch anthropic numpy scipy openneuro-py

Download data (one-time, ~4 GB for subject 1):
  openneuro download --dataset ds002376 --include sub-01
"""

import os
import json
import numpy as np
import anthropic
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("./data/broderick")   # where you put the downloaded data
SUBJECT     = "sub-01"
SFREQ_NEW   = 64          # downsample to this (paper uses 50, we use 64 for speed)
EPOCH_TMIN  = 0.0         # seconds after word onset
EPOCH_TMAX  = 0.8         # seconds after word onset
BANDPASS    = (1.0, 40.0) # Hz
TOP_K       = 5           # candidates per word sent to LLM
VOCAB_SIZE  = 250         # most-frequent words to retrieve from
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


# ── Step 1: Load EEG ──────────────────────────────────────────────────────────

def load_broderick(data_dir: Path, subject: str):
    """
    Load Broderick EEG + stimulus word timings.
    Returns raw MNE object and word event list.
    
    Data structure (after openneuro download):
        data_dir/
          sub-01/eeg/sub-01_task-audiobook_eeg.fif   (or .set)
          sub-01/eeg/sub-01_task-audiobook_events.tsv
    """
    # Try common file extensions
    for ext in ["_eeg.fif", "_task-audiobook_eeg.set", "_eeg.set"]:
        eeg_path = data_dir / subject / "eeg" / f"{subject}{ext}"
        if eeg_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No EEG file found in {data_dir / subject / 'eeg'}.\n"
            f"Download with: openneuro download --dataset ds002376 --include {subject}"
        )

    print(f"Loading EEG from {eeg_path}")
    if eeg_path.suffix == ".fif":
        raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)

    # Load word events
    events_path = data_dir / subject / "eeg" / f"{subject}_task-audiobook_events.tsv"
    words, onsets_sec = [], []
    if events_path.exists():
        import csv
        with open(events_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("trial_type", "").startswith("word"):
                    words.append(row.get("value", row.get("word", "")))
                    onsets_sec.append(float(row["onset"]))
    else:
        print("Warning: no events.tsv found — using dummy events for testing")
        # Dummy: one word per second for first 60 seconds
        words = [f"word{i}" for i in range(60)]
        onsets_sec = list(range(60))

    return raw, words, np.array(onsets_sec)


# ── Step 2: Preprocess EEG ────────────────────────────────────────────────────

def preprocess(raw: mne.io.BaseRaw, sfreq_new: int = SFREQ_NEW) -> mne.io.BaseRaw:
    print("Preprocessing EEG...")
    raw = raw.copy()
    raw.pick_types(eeg=True, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose=False)
    raw.resample(sfreq_new, verbose=False)
    return raw


def epoch_eeg(raw: mne.io.BaseRaw, onsets_sec: np.ndarray,
              tmin: float = EPOCH_TMIN, tmax: float = EPOCH_TMAX):
    """Cut fixed-length epochs around word onsets. Returns (N, channels, times)."""
    sfreq   = raw.info["sfreq"]
    data    = raw.get_data()          # (channels, total_samples)
    n_ch    = data.shape[0]
    n_times = int((tmax - tmin) * sfreq)
    epochs  = []
    good_idx = []

    for i, onset in enumerate(onsets_sec):
        start = int((onset + tmin) * sfreq)
        end   = start + n_times
        if end <= data.shape[1]:
            ep = data[:, start:end]
            # Z-score per channel for robustness
            ep = (ep - ep.mean(axis=1, keepdims=True)) / (ep.std(axis=1, keepdims=True) + 1e-8)
            epochs.append(ep)
            good_idx.append(i)

    return np.array(epochs), np.array(good_idx)  # (N, ch, time)


# ── Step 3: T5 Word Embeddings ────────────────────────────────────────────────

def get_t5_embeddings(words: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode words with T5-large middle layer (matches d'Ascoli paper).
    Returns (N_words, 1024).
    """
    print(f"Computing T5 embeddings for {len(words)} unique words...")
    from transformers import T5Tokenizer, T5EncoderModel
    import torch

    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model     = T5EncoderModel.from_pretrained("t5-large")
    model.eval()
    mid_layer = model.config.num_layers // 2

    unique_words = list(dict.fromkeys(words))  # preserve order, deduplicate
    word2emb: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in range(0, len(unique_words), batch_size):
            batch = unique_words[i : i + batch_size]
            enc   = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            out   = model(**enc, output_hidden_states=True)
            # Middle layer hidden states, mean-pool over tokens
            hs    = out.hidden_states[mid_layer]            # (B, seq, 1024)
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            embs  = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # (B, 1024)
            for word, emb in zip(batch, embs.cpu().numpy()):
                word2emb[word] = emb

    return np.array([word2emb[w] for w in words]), word2emb


# ── Step 4: Ridge Regression ──────────────────────────────────────────────────

def flatten_epochs(epochs: np.ndarray) -> np.ndarray:
    """(N, ch, time) → (N, ch*time) feature matrix."""
    return epochs.reshape(len(epochs), -1)


def train_ridge(X_train: np.ndarray, Y_train: np.ndarray):
    """Train ridge regression: EEG features → T5 embeddings."""
    print(f"Training ridge regression: X={X_train.shape} → Y={Y_train.shape}")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train)
    ridge  = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge.fit(X_s, Y_train)
    print(f"  Best alpha: {ridge.alpha_:.3f}")
    return ridge, scaler


def predict_embeddings(ridge, scaler, X: np.ndarray) -> np.ndarray:
    return ridge.predict(scaler.transform(X))


# ── Step 5: Vocabulary Retrieval ──────────────────────────────────────────────

def build_vocab(word2emb: dict, words_in_data: list[str], vocab_size: int = VOCAB_SIZE):
    """
    Build reduced vocabulary: top-N most frequent words that have embeddings.
    Returns (vocab_words, vocab_embs_normed).
    """
    from collections import Counter
    freq    = Counter(words_in_data)
    top_words = [w for w, _ in freq.most_common(vocab_size) if w in word2emb]
    vocab_embs = np.array([word2emb[w] for w in top_words])
    # L2-normalise for cosine similarity
    norms  = np.linalg.norm(vocab_embs, axis=1, keepdims=True) + 1e-8
    return top_words, vocab_embs / norms


def retrieve_top_k(pred_embs: np.ndarray, vocab_words: list, vocab_embs_normed: np.ndarray,
                   k: int = TOP_K) -> list[list[str]]:
    """
    For each predicted embedding, return top-K vocabulary words.
    pred_embs: (N, d)
    Returns list of N lists, each with K word strings.
    """
    # Normalise predictions
    norms = np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-8
    pred_normed = pred_embs / norms                              # (N, d)
    scores = pred_normed @ vocab_embs_normed.T                   # (N, V)
    top_k_idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return [[vocab_words[i] for i in row] for row in top_k_idx]


# ── Step 6: LLM Sentence Reconstruction ──────────────────────────────────────

def llm_reconstruct_sentence(
    top_k_per_word: list[list[str]],
    true_sentence:  str = "",
    client:         anthropic.Anthropic = None,
) -> str:
    """
    Send top-K candidates per word position to Claude.
    Returns the LLM's best guess at the full sentence.

    This is the novel contribution: the paper stops at retrieval.
    We use an LLM to pick the most coherent path through the candidates.
    """
    if client is None:
        client = anthropic.Anthropic()

    # Format the candidate grid for the LLM
    lines = []
    for i, candidates in enumerate(top_k_per_word):
        lines.append(f"  Position {i+1}: {' | '.join(candidates)}")
    candidate_grid = "\n".join(lines)

    prompt = f"""You are helping reconstruct a sentence from EEG brain signals.

A neural decoder has produced a list of top-{TOP_K} candidate words for each position in a sentence. The first candidate is the highest-confidence prediction from the brain signal; the others are alternatives.

Candidates (position: option1 | option2 | ... ):
{candidate_grid}

Task: Pick exactly one word per position to form the most grammatically correct and semantically coherent English sentence. You must use exactly one word from each position's candidates.

Respond with ONLY a JSON object, nothing else:
{{
  "sentence": "the reconstructed sentence here",
  "words_chosen": ["word1", "word2", ...]
}}"""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    import re
    raw = re.sub(r"^```(?:json)?\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)
    return data.get("sentence", ""), data.get("words_chosen", [])


# ── Evaluation ────────────────────────────────────────────────────────────────

def top_k_accuracy(top_k_per_word: list[list[str]], true_words: list[str]) -> float:
    correct = sum(t in cands for t, cands in zip(true_words, top_k_per_word))
    return correct / len(true_words) if true_words else 0.0


def word_error_rate(pred_sentence: str, true_sentence: str) -> float:
    pred  = pred_sentence.lower().split()
    true  = true_sentence.lower().split()
    # Simple WER via edit distance
    d = np.zeros((len(true) + 1, len(pred) + 1), dtype=int)
    for i in range(len(true) + 1): d[i][0] = i
    for j in range(len(pred) + 1): d[0][j] = j
    for i in range(1, len(true) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if true[i-1] == pred[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(true)][len(pred)] / max(len(true), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NeuroSpeak — Simple EEG + LLM Pipeline")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    raw, words, onsets_sec = load_broderick(DATA_DIR, SUBJECT)
    raw = preprocess(raw)
    epochs, good_idx = epoch_eeg(raw, onsets_sec)
    words = [words[i] for i in good_idx]

    print(f"Loaded {len(epochs)} epochs, {epochs.shape[1]} channels, "
          f"{epochs.shape[2]} timepoints each")

    # ── T5 embeddings ──────────────────────────────────────────────────────
    word_embs, word2emb = get_t5_embeddings(words)

    # ── Train/test split (80/20, no shuffle — temporal order matters) ──────
    n        = len(epochs)
    n_train  = int(0.8 * n)
    X        = flatten_epochs(epochs)
    Y        = word_embs

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    words_test      = words[n_train:]

    # ── Train ridge ────────────────────────────────────────────────────────
    ridge, scaler = train_ridge(X_train, Y_train)

    # ── Predict on test ────────────────────────────────────────────────────
    Y_pred = predict_embeddings(ridge, scaler, X_test)

    # ── Build vocab & retrieve ─────────────────────────────────────────────
    vocab_words, vocab_embs_normed = build_vocab(word2emb, words, VOCAB_SIZE)
    print(f"Vocabulary: {len(vocab_words)} words")

    top_k_all = retrieve_top_k(Y_pred, vocab_words, vocab_embs_normed, k=TOP_K)

    # ── Evaluate retrieval ─────────────────────────────────────────────────
    acc = top_k_accuracy(top_k_all, words_test)
    print(f"\nTop-{TOP_K} retrieval accuracy: {acc:.1%}  (chance = {TOP_K/VOCAB_SIZE:.1%})")

    # ── LLM reconstruction on a few test sentences ─────────────────────────
    print("\n" + "─" * 60)
    print("LLM sentence reconstruction (sample sentences)")
    print("─" * 60)

    client = anthropic.Anthropic()

    # Group test words into sentences of ~8 words for demo
    SENT_LEN = 8
    n_sents  = min(5, len(words_test) // SENT_LEN)

    results = []
    for s in range(n_sents):
        start = s * SENT_LEN
        end   = start + SENT_LEN
        true_sentence  = " ".join(words_test[start:end])
        top_k_sentence = top_k_all[start:end]

        pred_sentence, chosen_words = llm_reconstruct_sentence(
            top_k_sentence, true_sentence, client
        )
        wer = word_error_rate(pred_sentence, true_sentence)

        results.append({
            "true":      true_sentence,
            "predicted": pred_sentence,
            "wer":       wer,
        })

        print(f"\nSentence {s+1}:")
        print(f"  True:      {true_sentence}")
        print(f"  Predicted: {pred_sentence}")
        print(f"  WER:       {wer:.1%}")

    avg_wer = np.mean([r["wer"] for r in results])
    print(f"\nAverage WER across {n_sents} sentences: {avg_wer:.1%}")
    print(f"Top-{TOP_K} retrieval accuracy:          {acc:.1%}")

    # ── Save results ───────────────────────────────────────────────────────
    out = {
        "top_k_accuracy": acc,
        "avg_wer":        avg_wer,
        "vocab_size":     len(vocab_words),
        "n_test_words":   len(words_test),
        "sentences":      results,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to results.json")


# ── Demo mode (no real data needed) ──────────────────────────────────────────

def demo_llm_only():
    """
    Run just the LLM reconstruction step with fake retrieval output.
    Useful for testing the LLM integration without EEG data.
    """
    print("Running LLM-only demo (no EEG data needed)\n")

    # Simulate what the ridge regression would output for a sentence
    # from The Old Man and the Sea
    fake_top_k = [
        ["the",   "a",      "his",    "an",    "this"],
        ["old",   "young",  "big",    "small", "dark"],
        ["man",   "boy",    "fish",   "sea",   "boat"],
        ["had",   "was",    "went",   "came",  "sat"],
        ["gone",  "been",   "left",   "moved", "stayed"],
        ["far",   "deep",   "out",    "away",  "long"],
        ["into",  "across", "toward", "over",  "through"],
        ["the",   "a",      "that",   "this",  "his"],
        ["sea",   "water",  "ocean",  "river", "bay"],
    ]

    client = anthropic.Anthropic()
    sentence, words_chosen = llm_reconstruct_sentence(fake_top_k, client=client)

    print("Top-K candidates per position:")
    for i, cands in enumerate(fake_top_k):
        print(f"  {i+1}: {' | '.join(cands)}")

    print(f"\nLLM reconstructed: '{sentence}'")
    print(f"Words chosen: {words_chosen}")
    print(f"\nTrue sentence:     'the old man had gone far into the sea'")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv or not DATA_DIR.exists():
        print("Data directory not found — running LLM demo mode.")
        print("For full pipeline: download Broderick data and set DATA_DIR.\n")
        demo_llm_only()
    else:
        main()
