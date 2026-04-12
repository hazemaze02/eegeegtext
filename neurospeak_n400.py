"""
neurospeak_n400.py
==================
Full NeuroSpeak pipeline using synthetic EEG + N400 features + LLM reconstruction.

Run from the repo root:
    python neurospeak_n400.py

Requires:
    pip install google-genai numpy scipy scikit-learn matplotlib
    (transformers optional — uses simple embeddings if not installed)

Pipeline:
    1. Generate synthetic EEG with realistic N400 signatures
    2. Extract N400 + theta features  (teammate's n400_pipeline.py)
    3. Predict word embeddings  (ridge regression)
    4. Retrieve top-K candidates per word  (cosine similarity)
    5. LLM reconstructs the sentence  (Gemini API)
    6. Visualize N400 per decoded word
"""

import os
import json
import re
import warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from google import genai

# ── Import teammate's N400 module ─────────────────────────────────────────────
try:
    from sentence_decoding.n400_pipeline import (
        extract_n400_features,
        visualize_n400_per_word,
        run_n400_ablation_evaluation,
        compute_n400_from_batch,
    )
    print("[OK] Loaded n400_pipeline from sentence_decoding package")
except ImportError:
    try:
        from n400_pipeline import (
            extract_n400_features,
            visualize_n400_per_word,
            run_n400_ablation_evaluation,
            compute_n400_from_batch,
        )
        print("[OK] Loaded n400_pipeline from local directory")
    except ImportError:
        raise ImportError(
            "Could not find n400_pipeline.py.\n"
            "Make sure it is in sentence_decoding/ or the same directory as this script."
        )

# ── Config ────────────────────────────────────────────────────────────────────

SFREQ        = 50       # Hz — matches the paper's resample target
N_CHANNELS   = 64       # electrodes
EPOCH_SECS   = 3.0      # seconds per epoch (paper uses 3s windows)
N_TIMES      = int(EPOCH_SECS * SFREQ)   # = 150 time points
EMBED_DIM    = 128      # embedding dimension (simplified; paper uses T5's 1024)
TOP_K        = 5        # candidates per word sent to LLM
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GEMINI_MODEL = "gemini-2.5-flash"

# N400 channel layout — standard 10-20 names
CHANNEL_NAMES = (
    ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
     "FC5", "FC1", "FC2", "FC6",
     "T7", "C3", "Cz", "C4", "T8",
     "CP5", "CP1", "CP2", "CP4",        # <-- N400-rich channels
     "P7", "P3", "Pz", "P4", "P8",
     "POz", "O1", "Oz", "O2"]
    + [f"EXT{i}" for i in range(N_CHANNELS - 29)]
)

# ── Sample corpus ─────────────────────────────────────────────────────────────
# Short sentences from a naturalistic reading paradigm.
# "Expected" words have low N400; "unexpected" ones have high N400.

CORPUS = [
    # (sentence_words, expected_mask)
    # expected_mask: True = expected word (low N400), False = unexpected (high N400)
    (["the", "boy", "ate", "the", "cake"],
     [True,  True,  True,  True,  True]),

    (["she", "drank", "cold", "water", "quickly"],
     [True,  True,   True,   True,    True]),

    (["the", "cat", "sat", "on", "the", "mat"],
     [True,  True,  True,  True, True,  True]),

    (["he", "drove", "his", "car", "to", "work"],
     [True, True,    True,  True,  True, True]),

    # Sentences with unexpected endings (high N400 on last word)
    (["the", "pizza", "was", "too", "hot", "to", "cry"],
     [True,  True,    True,  True,  True,  True, False]),  # "cry" unexpected

    (["she", "watered", "her", "plants", "with", "gasoline"],
     [True,  True,      True,  True,     True,   False]),   # "gasoline" unexpected

    (["the", "surgeon", "operated", "with", "a", "banana"],
     [True,  True,      True,       True,   True, False]),  # "banana" unexpected

    (["he", "wrote", "the", "letter", "with", "his", "elbow"],
     [True, True,    True,  True,     True,   True,  False]), # "elbow" unexpected
]

# Full vocabulary (all unique words in corpus + distractors)
DISTRACTORS = [
    "dog", "run", "fast", "red", "blue", "house", "tree", "book",
    "fish", "bird", "rain", "sun", "moon", "star", "hand", "door",
    "eat", "drink", "sleep", "walk", "talk", "read", "write", "play",
    "big", "small", "hot", "cold", "old", "new", "good", "bad",
]

VOCAB = sorted(set(
    w for sent, _ in CORPUS for w in sent
) | set(DISTRACTORS))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Word embeddings (simplified; replace with T5 if available)
# ══════════════════════════════════════════════════════════════════════════════

def make_word_embeddings(vocab: list[str], dim: int = EMBED_DIM,
                         seed: int = 42) -> dict[str, np.ndarray]:
    """
    Create consistent pseudo-random embeddings per word.
    In production these come from T5-large middle layer (d'Ascoli et al.).
    Here we use seeded random vectors so similar words get similar embeddings.
    """
    rng = np.random.default_rng(seed)
    embs = {}
    for word in vocab:
        # Seed per word so the same word always gets the same embedding
        word_seed = int.from_bytes(word.encode()[:4].ljust(4, b"\x00"), "little")
        w_rng = np.random.default_rng(word_seed % (2**31))
        embs[word] = w_rng.normal(0, 1, dim).astype(np.float32)
        # L2 normalise
        embs[word] /= np.linalg.norm(embs[word]) + 1e-8
    return embs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Synthetic EEG generation with realistic N400 signatures
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_eeg(
    words:         list[str],
    expected_mask: list[bool],
    n_channels:    int   = N_CHANNELS,
    sfreq:         float = SFREQ,
    n_times:       int   = N_TIMES,
    noise_std:     float = 10.0,   # µV background noise
    n400_std:      float = 5.0,    # µV — N400 component amplitude
    seed:          int   = 0,
) -> np.ndarray:
    """
    Generate (n_words, n_channels, n_times) EEG with realistic N400.

    For unexpected words (expected_mask=False):
      - Add a negative deflection at centro-parietal channels
        peaking at 400ms (sample 20 at 50Hz), width ~100ms

    For expected words:
      - Background EEG noise only
    """
    rng    = np.random.default_rng(seed)
    n_words = len(words)
    epochs = rng.normal(0, noise_std, (n_words, n_channels, n_times)).astype(np.float32)

    # N400 channel indices (centro-parietal in our layout)
    n400_ch_idx = [i for i, name in enumerate(CHANNEL_NAMES[:n_channels])
                   if name in {"CP1", "CP2", "CP4", "CP5", "Pz", "Cz", "P3", "P4"}]
    if not n400_ch_idx:
        n400_ch_idx = list(range(n_channels // 2, n_channels // 2 + 8))

    # Build Gaussian N400 waveform centred at 400ms
    peak_sample = int(0.400 * sfreq)   # sample 20 at 50Hz
    width       = int(0.080 * sfreq)   # ~80ms std
    t           = np.arange(n_times)
    n400_kernel = np.exp(-0.5 * ((t - peak_sample) / max(width, 1)) ** 2)
    n400_kernel /= n400_kernel.max() + 1e-8

    for i, (word, expected) in enumerate(zip(words, expected_mask)):
        if not expected:
            # Unexpected word → large negative N400 on centro-parietal channels
            amplitude = rng.uniform(n400_std * 0.8, n400_std * 1.5)
            for ch in n400_ch_idx:
                epochs[i, ch] -= amplitude * n400_kernel   # negative deflection

        # Add small alpha oscillation (8–12 Hz) for realism
        alpha_freq = rng.uniform(9, 11)
        alpha_amp  = rng.uniform(2, 5)
        t_sec      = np.linspace(0, EPOCH_SECS, n_times)
        alpha_wave = alpha_amp * np.sin(2 * np.pi * alpha_freq * t_sec)
        epochs[i] += alpha_wave[np.newaxis, :]

    return epochs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EEG feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(epochs: np.ndarray) -> np.ndarray:
    """
    Combine:
      - Flattened raw EEG  (n_channels × n_times)
      - N400 amplitude + theta power  (from teammate's module)

    Returns (n_epochs, n_features) feature matrix.
    """
    # Raw EEG features
    raw_feats = epochs.reshape(len(epochs), -1)   # (N, ch*time)

    # N400 features from teammate's module: (N, 2) — [n400, theta]
    n400_feats = compute_n400_from_batch(
        epochs,
        channel_names=CHANNEL_NAMES[:epochs.shape[1]],
        sfreq=SFREQ,
    )

    return np.hstack([raw_feats, n400_feats])     # (N, ch*time + 2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Ridge regression: EEG features → word embeddings
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(X_train, Y_train, X_test):
    print(f"Training ridge regression: {X_train.shape} → {Y_train.shape}")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(5, len(X_train)))
    ridge.fit(X_tr, Y_train)
    print(f"  Best alpha: {ridge.alpha_:.3f}")

    return ridge.predict(X_te)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Vocabulary retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_top_k(pred_embs: np.ndarray,
                   vocab_words: list[str],
                   vocab_embs:  np.ndarray,
                   k: int = TOP_K) -> list[list[str]]:
    """Cosine similarity retrieval. Returns top-K words per prediction."""
    # Normalise predictions
    norms       = np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-8
    pred_normed = pred_embs / norms
    scores      = pred_normed @ vocab_embs.T          # (N, V)
    top_k_idx   = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return [[vocab_words[i] for i in row] for row in top_k_idx]


def top_k_accuracy(top_k_lists: list[list[str]], true_words: list[str]) -> float:
    hits = sum(t in cands for t, cands in zip(true_words, top_k_lists))
    return hits / len(true_words) if true_words else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LLM sentence reconstruction (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def llm_reconstruct(
    top_k_per_word: list[list[str]],
    n400_values:    np.ndarray,
    client:         genai.Client,
) -> tuple[str, list[str]]:
    """
    Send top-K candidates + N400 surprise signal to Gemini.
    N400 values tell the LLM which positions were semantically unexpected,
    guiding it to be more conservative (pick common words) at high-N400 spots.
    """
    lines = []
    for i, (cands, n400) in enumerate(zip(top_k_per_word, n400_values)):
        surprise = (
            "HIGH surprise (unexpected word — pick carefully)"
            if n400 > 0.5
            else "low surprise (expected word)"
        )
        lines.append(f"  Position {i+1} [{surprise}]: {' | '.join(cands)}")

    candidate_block = "\n".join(lines)

    prompt = f"""You are a neural sentence decoder. You receive EEG brain signal data processed into word candidates.

Each position shows the top-{TOP_K} candidate words decoded from brain signals, plus an N400 surprise indicator:
- LOW surprise = the brain found this word easy to predict (use the first candidate)
- HIGH surprise = the brain found this word unexpected (consider all candidates carefully)

Candidates:
{candidate_block}

Reconstruct the most grammatically correct and semantically coherent English sentence.
Use exactly one word per position from its candidates list.

Respond ONLY with valid JSON, no other text:
{{"sentence": "your reconstructed sentence", "words": ["word1", "word2", ...]}}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    raw  = response.text.strip()
    raw  = re.sub(r"^```(?:json)?\n?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    return data.get("sentence", ""), data.get("words", [])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("NeuroSpeak — N400-augmented EEG + LLM Sentence Reconstruction")
    print("=" * 65)

    # ── Build vocab embeddings ──────────────────────────────────────────
    print(f"\n[1] Building embeddings for {len(VOCAB)} words...")
    word2emb = make_word_embeddings(VOCAB, dim=EMBED_DIM)
    vocab_words = list(word2emb.keys())
    vocab_embs  = np.array([word2emb[w] for w in vocab_words])
    # L2 normalise vocab for cosine retrieval
    vocab_embs /= np.linalg.norm(vocab_embs, axis=1, keepdims=True) + 1e-8

    # ── Generate synthetic EEG corpus ───────────────────────────────────
    print("[2] Generating synthetic EEG with N400 signatures...")
    all_epochs, all_words, all_expected = [], [], []
    for sent_words, sent_expected in CORPUS:
        epochs = generate_synthetic_eeg(sent_words, sent_expected, seed=len(all_words))
        all_epochs.append(epochs)
        all_words.extend(sent_words)
        all_expected.extend(sent_expected)

    epochs_all  = np.concatenate(all_epochs, axis=0)  # (N_total, ch, time)
    n_total     = len(all_words)
    print(f"   Total epochs: {n_total}  |  shape: {epochs_all.shape}")

    # ── Extract features (raw EEG + N400 from teammate's module) ────────
    print("[3] Extracting EEG + N400 features...")
    X = extract_features(epochs_all)                   # (N, ch*time + 2)
    Y = np.array([word2emb[w] for w in all_words])    # (N, embed_dim)
    print(f"   Feature matrix: {X.shape}  →  Target: {Y.shape}")

    # ── Train / test split (first 80% train, last 20% test) ─────────────
    split   = int(0.8 * n_total)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    words_test      = all_words[split:]
    expected_test   = all_expected[split:]

    # ── Train ridge regression ───────────────────────────────────────────
    print("[4] Training ridge regression...")
    Y_pred = train_and_predict(X_train, Y_train, X_test)

    # ── Vocabulary retrieval ─────────────────────────────────────────────
    print("[5] Retrieving top-K candidates...")
    top_k_all = retrieve_top_k(Y_pred, vocab_words, vocab_embs, k=TOP_K)
    acc       = top_k_accuracy(top_k_all, words_test)
    chance    = TOP_K / len(vocab_words)
    print(f"   Top-{TOP_K} accuracy: {acc:.1%}  (chance = {chance:.1%})")

    # ── N400 features for test set ───────────────────────────────────────
    n400_test_feats = extract_n400_features(
        epochs_all[split:],
        channel_names=CHANNEL_NAMES[:N_CHANNELS],
        sfreq=SFREQ,
    )
    n400_vals = n400_test_feats["n400"]

    # ── LLM reconstruction ───────────────────────────────────────────────
    print("\n[6] LLM sentence reconstruction...")
    print("─" * 65)

    client   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results  = []
    sent_len = 5   # decode 5-word chunks

    # Group test words into sentences
    n_sents = max(1, len(words_test) // sent_len)

    for s in range(n_sents):
        start = s * sent_len
        end   = min(start + sent_len, len(words_test))

        true_sentence  = " ".join(words_test[start:end])
        top_k_sentence = top_k_all[start:end]
        n400_sentence  = n400_vals[start:end]

        pred_sentence, chosen = llm_reconstruct(top_k_sentence, n400_sentence, client)

        # Word error rate
        pred_words = pred_sentence.lower().split()
        true_words = true_sentence.lower().split()
        hits = sum(p == t for p, t in zip(pred_words, true_words))
        word_acc = hits / max(len(true_words), 1)

        results.append({
            "true":      true_sentence,
            "predicted": pred_sentence,
            "word_acc":  word_acc,
            "n400":      n400_sentence.tolist(),
        })

        print(f"\nSentence {s+1}:")
        print(f"  True:       {true_sentence}")
        print(f"  Predicted:  {pred_sentence}")
        print(f"  Word acc:   {word_acc:.1%}")
        print(f"  N400 vals:  {[f'{v:.2f}' for v in n400_sentence]}")
        unexpected = [w for w, e in zip(words_test[start:end], expected_test[start:end]) if not e]
        if unexpected:
            print(f"  Unexpected: {unexpected}  ← should have high N400")

    avg_acc = np.mean([r["word_acc"] for r in results])
    print(f"\n{'─'*65}")
    print(f"Average word accuracy: {avg_acc:.1%}")
    print(f"Top-{TOP_K} retrieval accuracy: {acc:.1%}  (chance {chance:.1%})")

    # ── N400 visualization ───────────────────────────────────────────────
    print("\n[7] Generating N400 visualization...")
    # Use first full sentence from test set for visualization
    viz_end = min(sent_len, len(words_test))
    visualize_n400_per_word(
        words=words_test[:viz_end],
        n400_values=n400_vals[:viz_end],
        output_path="n400_visualization.png",
    )

    # ── N400 ablation comparison ─────────────────────────────────────────
    # Train a second model WITHOUT N400 features for comparison
    print("\n[8] Running N400 ablation...")
    X_raw = X[:, :-2]   # drop the 2 N400 columns
    Y_pred_no_n400 = train_and_predict(X_raw[:split], Y_train, X_raw[split:])
    top_k_no_n400  = retrieve_top_k(Y_pred_no_n400, vocab_words, vocab_embs)
    acc_no_n400    = top_k_accuracy(top_k_no_n400, words_test)

    run_n400_ablation_evaluation(
        bleu_with_n400=acc,
        bleu_without_n400=acc_no_n400,
        words=words_test,
        n400_values=n400_vals,
        n_gram=1,
    )

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "top_k_accuracy_with_n400":    acc,
        "top_k_accuracy_without_n400": acc_no_n400,
        "n400_delta":                  acc - acc_no_n400,
        "avg_word_accuracy":           avg_acc,
        "vocab_size":                  len(vocab_words),
        "n_test_words":                len(words_test),
        "sentences":                   results,
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n[Done] Outputs:")
    print("  results.json           — accuracy metrics + decoded sentences")
    print("  n400_visualization.png — N400 per word color visualization")


if __name__ == "__main__":
    main()
