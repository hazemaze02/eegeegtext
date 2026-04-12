"""
n400_pipeline.py — N400-augmented EEG-to-Text Decoding Pipeline
================================================================

This module extends the sentence_decoding pipeline with N400 features.

BACKGROUND — WHAT IS N400?
---------------------------
The N400 is a negative voltage deflection in the scalp EEG that peaks around
400 ms after a word is shown. It was first described by Kutas & Hillyard (1980)
and is one of the most robustly replicated findings in cognitive neuroscience.

  • Large (more negative) N400  →  word was hard to integrate semantically
    (e.g. "The pizza was too hot to CRY")
  • Small (near-zero) N400     →  word was easy to predict from context
    (e.g. "The pizza was too hot to EAT")

N400 is maximal at *centro-parietal* electrodes (Cz, Pz, CP1/CP2, etc.)
and is typically measured in the 300–500 ms post-word window.

We extract N400 amplitude per word as a proxy for "how well the model (and
the participant) understood that word in context."  A second feature —
theta-band (4–8 Hz) power — is extracted as a complementary cognitive-load
marker (theta increases with working memory load).

HOW THESE FEATURES ARE USED
----------------------------
Both scalars are concatenated to the brain embedding vector produced by the
EEG encoder (SimpleConvTimeAgg), and then projected back to the original
embedding dimension via a small linear layer.  This is controlled by a
`use_n400` flag in BrainModule so the addition can be ablated at evaluation.

NOTE ON FAKE DATA
-----------------
TestEeg2024 is random Gaussian noise, so N400 values will also be random.
Any performance differences with/without N400 on the fake dataset are due to
chance.  In real EEG recordings from participants reading sentences, N400
amplitude correlates with word predictability, and we would expect these
features to provide meaningful signal to the decoder.

USAGE (standalone)
------------------
    from sentence_decoding.n400_pipeline import (
        extract_n400_features,
        visualize_n400_per_word,
        run_n400_ablation_evaluation,
        compute_n400_word_freq_correlation,
    )

    # Example: batch of EEG epochs from the dataloader
    import numpy as np
    fake_epochs = np.random.randn(32, 64, 150)   # 32 epochs, 64 ch, 150 time-pts
    channel_names = [f"ch{i}" for i in range(64)]

    feats = extract_n400_features(fake_epochs, channel_names=channel_names)
    print(feats['n400'].shape)   # (32,)
    print(feats['theta'].shape)  # (32,)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import warnings
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")        # Non-interactive backend — safe for scripts/servers
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import signal
from scipy.stats import pearsonr, zscore

# ---------------------------------------------------------------------------
# Centro-parietal channels where N400 is maximal (standard 10-20 naming).
# If none of these are present in the dataset, all channels are used instead.
# ---------------------------------------------------------------------------
N400_CHANNELS = [
    # Midline central-parietal
    "Cz", "CPz", "Pz",
    # Lateral central-parietal
    "CP1", "CP2", "CP3", "CP4",
    # Adjacent central & parietal
    "C3", "C4",
    "P3", "P4",
]


# ===========================================================================
# TASK 1 — N400 FEATURE EXTRACTION
# ===========================================================================

def extract_n400_features(
    epochs: np.ndarray,
    channel_names: Optional[List[str]] = None,
    sfreq: float = 50.0,
    n400_tmin: float = 0.300,
    n400_tmax: float = 0.500,
) -> Dict[str, np.ndarray]:
    """
    Extract N400 amplitude and theta power from EEG epochs.

    The N400 is computed as the mean EEG voltage in the 300–500 ms window
    on centro-parietal channels.  Theta power (4–8 Hz) is computed via
    Welch's PSD method across the same channels.

    Both features are z-score normalized across the batch so they are on a
    comparable scale to the brain embedding vectors they will be concatenated
    to.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_timepoints)
        EEG epochs aligned to word onset.  Time 0 is word presentation.
        In the current pipeline this data is already resampled to `sfreq`
        Hz and baseline-corrected by the Eeg feature class.

    channel_names : list of str or None
        Electrode labels in the same order as the channel dimension of
        `epochs`.  Common names: ["Fp1", "Fp2", ..., "Cz", "Pz", ...].
        If None or if no centro-parietal channels are found, all channels
        are used as fallback (appropriate for the fake dataset).

    sfreq : float
        Sampling frequency of `epochs` in Hz.
        Default 50.0 Hz — the pipeline's target resample frequency.

        NOTE: In real data recorded at e.g. 1000 Hz, pass sfreq=1000.
        The N400 window computation is frequency-agnostic (it converts ms
        to samples using this value).

    n400_tmin, n400_tmax : float
        Start / end of the N400 window in seconds post word onset.
        Standard: 0.300–0.500 s.  Adjust for your experiment's SOA.

    Returns
    -------
    dict with keys:
        'n400'            : np.ndarray (n_epochs,)   — z-scored N400 amplitude
        'theta'           : np.ndarray (n_epochs,)   — z-scored theta power
        'n400_raw'        : np.ndarray (n_epochs,)   — raw (µV) N400 amplitude
        'theta_raw'       : np.ndarray (n_epochs,)   — raw theta band power
        'channel_indices' : list[int]                 — channels used for N400

    Notes on the fake dataset
    -------------------------
    TestEeg2024 generates random Gaussian noise at ~10 µV, so N400 values
    will be random scalars.  In a real experiment, words that are hard to
    predict (high cloze uncertainty) would show systematically larger N400
    amplitudes.  The normalization and shapes are correct regardless of
    whether the underlying signal is real or random.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must be 3-D (n_epochs, n_channels, n_timepoints), "
            f"got shape {epochs.shape}"
        )
    n_epochs, n_channels, n_timepoints = epochs.shape

    # ------------------------------------------------------------------
    # Step 1: Identify which channels to use for N400 computation
    # ------------------------------------------------------------------
    channel_indices = _find_n400_channels(channel_names, n_channels)

    # ------------------------------------------------------------------
    # Step 2: Convert the N400 time window (seconds) to sample indices
    # ------------------------------------------------------------------
    # At 50 Hz:  300 ms → sample 15,  500 ms → sample 25
    t_start = int(n400_tmin * sfreq)
    t_end   = int(n400_tmax * sfreq)

    # Guard against epochs that are shorter than the N400 window
    t_start = max(0, min(t_start, n_timepoints - 2))
    t_end   = max(t_start + 1, min(t_end, n_timepoints))

    if t_end - t_start < 2:
        warnings.warn(
            f"[N400] N400 window [{n400_tmin}s – {n400_tmax}s] maps to only "
            f"{t_end - t_start} sample(s) at {sfreq} Hz.  "
            f"Epoch length = {n_timepoints} samples.  "
            f"Using the first 10 samples as emergency fallback."
        )
        t_start, t_end = 0, min(10, n_timepoints)

    # ------------------------------------------------------------------
    # Step 3: Compute N400 — mean amplitude in the 300–500 ms window
    # ------------------------------------------------------------------
    # Slice to N400 channels and the relevant time window.
    # Shape: (n_epochs, n_n400_channels, n_window_samples)
    n400_window = epochs[:, channel_indices, t_start:t_end]

    # Average across both the channel dimension and the time dimension.
    # Result: one scalar per epoch (the "N400 amplitude").
    # Convention: more negative values = larger N400 response.
    n400_raw = n400_window.mean(axis=(1, 2))   # shape: (n_epochs,)

    # ------------------------------------------------------------------
    # Step 4: Compute theta power (4–8 Hz) via Welch's PSD
    # ------------------------------------------------------------------
    # Theta oscillations (4–8 Hz) are linked to working memory and cognitive
    # load.  Words that are harder to process tend to increase theta power.
    theta_raw = _compute_theta_power(
        epochs[:, channel_indices, :],  # only N400 channels
        sfreq=sfreq,
    )

    # ------------------------------------------------------------------
    # Step 5: Z-score normalize across all epochs in the batch
    # ------------------------------------------------------------------
    # This puts N400 and theta on a unit-variance, zero-mean scale,
    # making them numerically compatible with the brain embedding vectors.
    n400_z = _safe_zscore(n400_raw)
    theta_z = _safe_zscore(theta_raw)

    return {
        "n400":            n400_z,
        "theta":           theta_z,
        "n400_raw":        n400_raw,
        "theta_raw":       theta_raw,
        "channel_indices": channel_indices,
    }


def _find_n400_channels(
    channel_names: Optional[List[str]],
    n_channels: int,
) -> List[int]:
    """
    Return the channel indices to use for N400 computation.

    Priority order:
      1. Centro-parietal channels from N400_CHANNELS list (if found)
      2. All channels (fallback for unnamed or non-standard datasets)
    """
    if channel_names is None or len(channel_names) == 0:
        # No names available — typical for the fake dataset when passed raw
        print(
            "[N400] No channel names provided.  Using ALL channels as fallback.\n"
            "       In real data, provide standard 10-20 names so that only\n"
            "       centro-parietal channels (Cz, Pz, CPz, CP1, CP2) are used."
        )
        return list(range(n_channels))

    # Case-insensitive matching against the N400 channel list
    target_lower = {ch.lower() for ch in N400_CHANNELS}
    matched = [
        (i, name)
        for i, name in enumerate(channel_names)
        if name.strip().lower() in target_lower
    ]

    if not matched:
        print(
            f"[N400] None of the target channels {N400_CHANNELS}\n"
            f"       were found in the provided {len(channel_names)} channel names.\n"
            f"       Using ALL channels as fallback."
        )
        return list(range(n_channels))

    indices = [i for i, _ in matched]
    names   = [name for _, name in matched]
    print(
        f"[N400] Using {len(indices)} centro-parietal channels for N400: {names}"
    )
    return indices


def _compute_theta_power(
    epochs_subset: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute mean theta band (4–8 Hz) power for each epoch using Welch's PSD.

    Parameters
    ----------
    epochs_subset : np.ndarray, shape (n_epochs, n_channels, n_timepoints)
    sfreq : float

    Returns
    -------
    theta_power : np.ndarray, shape (n_epochs,)

    Implementation note
    -------------------
    scipy.signal.welch divides the signal into overlapping segments and
    averages their periodograms.  This gives a smoother PSD than a single
    FFT, which is important for short EEG epochs.
    """
    nyquist = sfreq / 2.0
    theta_low, theta_high = 4.0, 8.0

    # Safety check: if sfreq is too low, theta is beyond Nyquist
    if theta_high >= nyquist:
        warnings.warn(
            f"[N400] sfreq={sfreq} Hz → Nyquist={nyquist} Hz is below the "
            f"theta upper bound {theta_high} Hz.  Returning zero theta power.\n"
            f"       (This is expected only if you manually set a very low sfreq.)"
        )
        return np.zeros(epochs_subset.shape[0])

    n_epochs, n_channels, n_times = epochs_subset.shape

    # Number of samples per Welch segment — use 1-second window or full epoch
    nperseg = min(n_times, int(sfreq))
    if nperseg < 4:
        warnings.warn(
            f"[N400] Epoch too short ({n_times} samples) for PSD estimation.  "
            f"Returning zero theta power."
        )
        return np.zeros(n_epochs)

    theta_powers = np.zeros(n_epochs)
    for epoch_idx in range(n_epochs):
        # epochs_subset[epoch_idx] shape: (n_channels, n_times)
        freqs, psd = signal.welch(
            epochs_subset[epoch_idx],
            fs=sfreq,
            nperseg=nperseg,
            axis=-1,         # apply Welch along the time axis
        )
        # psd shape: (n_channels, n_freqs)
        # freqs shape: (n_freqs,)

        theta_mask = (freqs >= theta_low) & (freqs <= theta_high)
        if theta_mask.any():
            # Mean power across channels and theta-band frequencies
            theta_powers[epoch_idx] = psd[:, theta_mask].mean()
        # else stays 0 (no frequencies in band — shouldn't happen normally)

    return theta_powers


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalize an array.  If std ≈ 0 (e.g. all-zero signal),
    return the array unchanged to avoid division by zero.
    """
    std = x.std()
    if std < 1e-10:
        warnings.warn(
            "[N400] Standard deviation ≈ 0 — skipping z-score normalization.  "
            "This can happen with constant/silent EEG channels."
        )
        return x.copy()
    return zscore(x, nan_policy="omit")


# ===========================================================================
# TASK 3 — VISUALIZATION
# ===========================================================================

def visualize_n400_per_word(
    words: List[str],
    n400_values: np.ndarray,
    output_path: str = "n400_visualization.png",
    low_thresh: float = -0.5,
    high_thresh: float = 0.5,
) -> None:
    """
    Plot a sentence with each word color-coded by N400 amplitude.

    Color scheme:
      • Green  — low N400   (easy to process, well predicted by context)
      • Yellow — medium N400
      • Red    — high N400  (hard to process, unexpected / rare word)

    Also prints a text summary to the terminal, e.g.:
      The[low] cat[low] sat[low] on[low] quantum[HIGH] entanglement[HIGH]

    Parameters
    ----------
    words : list of str
        The decoded words (one per epoch).
    n400_values : array-like, shape (n_words,)
        Z-scored N400 amplitude per word.  Higher = harder to process.
    output_path : str
        Path to save the PNG visualization.
    low_thresh : float
        Words with n400 < low_thresh are colored green.
    high_thresh : float
        Words with n400 > high_thresh are colored red.
    """
    n400_values = np.asarray(n400_values, dtype=float)
    words = list(words)

    if len(words) != len(n400_values):
        raise ValueError(
            f"words ({len(words)}) and n400_values ({len(n400_values)}) "
            f"must have the same length."
        )
    if len(words) == 0:
        warnings.warn("[N400] Empty word list — nothing to visualize.")
        return

    # ------------------------------------------------------------------
    # Terminal text summary (always printed, even if matplotlib fails)
    # ------------------------------------------------------------------
    _print_n400_text(words, n400_values, low_thresh, high_thresh)

    # ------------------------------------------------------------------
    # Matplotlib figure
    # ------------------------------------------------------------------
    n_words = len(words)
    fig_width = max(10, n_words * 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    ax.set_xlim(-0.1, n_words + 0.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axis("off")

    ax.set_title(
        "N400 Amplitude per Word\n"
        "Green = easy to process (low N400)    |    "
        "Yellow = medium    |    "
        "Red = hard to process (high N400)",
        fontsize=10,
        pad=8,
    )

    for i, (word, n400) in enumerate(zip(words, n400_values)):
        color = _n400_to_color(n400, low_thresh, high_thresh)
        dark  = _is_dark_color(color)
        txt_color = "white" if dark else "#1a1a1a"

        # Colored word box
        rect = FancyBboxPatch(
            (i + 0.05, 0.30),
            0.88, 0.52,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)

        # Word text
        ax.text(
            i + 0.49, 0.59,
            word,
            ha="center", va="center",
            fontsize=max(7, min(11, 90 // n_words)),
            fontweight="bold",
            color=txt_color,
            zorder=3,
        )

        # N400 value (small, below word)
        ax.text(
            i + 0.49, 0.37,
            f"{n400:+.2f}",
            ha="center", va="center",
            fontsize=7,
            color=txt_color,
            zorder=3,
        )

    # Legend
    legend_items = [
        (f"< {low_thresh:.1f}  (low N400)", "#2ecc71"),
        (f"{low_thresh:.1f} – {high_thresh:.1f}  (medium)", "#f39c12"),
        (f"> {high_thresh:.1f}  (high N400)", "#e74c3c"),
    ]
    legend_x_start = 0.0
    block_w = n_words / max(len(legend_items), 1)
    for j, (label, color) in enumerate(legend_items):
        lx = legend_x_start + j * block_w
        rect = FancyBboxPatch(
            (lx + 0.1, 0.04), block_w - 0.2, 0.16,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="white",
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            lx + block_w / 2, 0.12,
            label,
            ha="center", va="center",
            fontsize=8,
            color="white",
            zorder=3,
        )

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[N400] Visualization saved → {output_path}")
    except Exception as e:
        warnings.warn(f"[N400] Could not save visualization to {output_path}: {e}")
    finally:
        plt.close(fig)


def _n400_to_color(n400_val: float, low_thresh: float, high_thresh: float) -> str:
    """
    Map a z-scored N400 value to an RGB hex color.

    Mapping:
      n400 ≤ low_thresh  → green  (#2ecc71)
      low < n400 < high  → interpolate green → yellow → red
      n400 ≥ high_thresh → red    (#e74c3c)
    """
    if n400_val <= low_thresh:
        return "#2ecc71"
    if n400_val >= high_thresh:
        return "#e74c3c"

    # Normalized position in [0, 1] within the mid range
    t = (n400_val - low_thresh) / (high_thresh - low_thresh + 1e-10)

    # Two-step interpolation: green(46,204,113) → yellow(243,156,18) → red(231,76,60)
    GREEN  = (46,  204, 113)
    YELLOW = (243, 156, 18)
    RED    = (231, 76,  60)

    if t < 0.5:
        # Green → Yellow
        s = t * 2
        r, g, b = [int(GREEN[k] + (YELLOW[k] - GREEN[k]) * s) for k in range(3)]
    else:
        # Yellow → Red
        s = (t - 0.5) * 2
        r, g, b = [int(YELLOW[k] + (RED[k] - YELLOW[k]) * s) for k in range(3)]

    return f"#{r:02x}{g:02x}{b:02x}"


def _is_dark_color(hex_color: str) -> bool:
    """Return True if a hex color is 'dark' (use white text on top of it)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    # Standard relative luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 140


def _print_n400_text(
    words: List[str],
    n400_values: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> None:
    """
    Print words with N400 labels to stdout.

    Example output:
        The[low] cat[low] sat[low] on[low] quantum[HIGH] entanglement[HIGH]
    """
    parts = []
    for word, n400 in zip(words, n400_values):
        if n400 <= low_thresh:
            label = "low"
        elif n400 >= high_thresh:
            label = "HIGH"
        else:
            label = "mid"
        parts.append(f"{word}[{label}]")

    print("\n[N400] Per-word N400 labels (z-scored amplitude):")
    print("  " + " ".join(parts))
    print(
        f"  Thresholds: low < {low_thresh:.1f} (green), "
        f"mid, high > {high_thresh:.1f} (red)\n"
    )


# ===========================================================================
# TASK 4 — EVALUATION
# ===========================================================================

def compute_n400_word_freq_correlation(
    words: List[str],
    n400_values: np.ndarray,
) -> Dict:
    """
    Sanity check: correlate N400 amplitude with word rarity.

    Expected relationship (from the literature):
        Rare / unexpected words  →  large (positive z-scored) N400
        Common / expected words  →  small (near-zero) N400

    This correlation should be positive if our N400 extraction is working
    correctly on real EEG data.  For random (fake) EEG, it will be near zero.

    Parameters
    ----------
    words : list of str
    n400_values : np.ndarray, shape (n_words,)
        Z-scored N400 values.

    Returns
    -------
    dict with keys:
        'correlation'  — Pearson r (rarity vs N400)
        'p_value'      — two-tailed p-value
        'note'         — explanation string
        'freq_source'  — which frequency source was used
    """
    n400_values = np.asarray(n400_values, dtype=float)

    # Try to use wordfreq for real frequency estimates (optional dependency)
    try:
        from wordfreq import word_frequency
        freqs = np.array([word_frequency(w.lower(), "en") + 1e-10 for w in words])
        # Word rarity: -log(frequency).  Rare words → high rarity → high N400.
        rarity = -np.log(freqs)
        freq_source = "wordfreq (log-frequency)"
    except ImportError:
        # Fallback: word length as rough rarity proxy.
        # In English, shorter words tend to be more frequent (Zipf's law).
        rarity = np.array([len(w) for w in words], dtype=float)
        freq_source = "word length (fallback proxy — install wordfreq for real freqs)"

    # Check we have enough variance to compute a meaningful correlation
    if len(np.unique(rarity)) < 2 or len(np.unique(n400_values)) < 2:
        result = {
            "correlation": 0.0,
            "p_value": 1.0,
            "note": (
                "Insufficient variance to compute correlation.  "
                "This is expected with fake/random EEG data."
            ),
            "freq_source": freq_source,
        }
        print("[N400] Sanity check: insufficient variance — skipping correlation.")
        return result

    r, p = pearsonr(rarity, n400_values)

    print("\n[N400] Sanity check — N400 vs word rarity correlation:")
    print(f"  Frequency source : {freq_source}")
    print(f"  Pearson r        : {r:+.3f}")
    print(f"  p-value          : {p:.4f}")

    if r > 0.2:
        print("  → Positive correlation: rarer words drive higher N400 ✓ (expected)")
    elif r < -0.2:
        print("  → Negative correlation: unexpected for real EEG data ✗")
    else:
        print(
            "  → Near-zero correlation (~expected for random/fake EEG data).\n"
            "     With real EEG, you would expect r ≈ 0.2–0.5."
        )

    return {
        "correlation": float(r),
        "p_value": float(p),
        "note": (
            f"Pearson r between word rarity ({freq_source}) and N400 amplitude."
        ),
        "freq_source": freq_source,
    }


def run_n400_ablation_evaluation(
    bleu_with_n400: float,
    bleu_without_n400: float,
    words: Optional[List[str]] = None,
    n400_values: Optional[np.ndarray] = None,
    n_gram: int = 1,
) -> Dict:
    """
    Print a formatted summary comparing decoding performance with/without N400.

    Parameters
    ----------
    bleu_with_n400 : float
        BLEU-n_gram score from the run with use_n400=True.
    bleu_without_n400 : float
        BLEU-n_gram score from the run with use_n400=False.
    words : list of str, optional
        Decoded words (used for the N400–frequency sanity check).
    n400_values : np.ndarray, optional
        Corresponding N400 values (used for the sanity check).
    n_gram : int
        Which BLEU order was measured (1 or 2).

    Returns
    -------
    dict with keys 'delta', 'relative_change_pct', 'helped', 'bleu_with', 'bleu_without'
    """
    delta      = bleu_with_n400 - bleu_without_n400
    rel_change = (delta / (abs(bleu_without_n400) + 1e-12)) * 100.0

    sep = "=" * 62
    print(f"\n{sep}")
    print("  N400 ABLATION EVALUATION SUMMARY")
    print(sep)
    print(f"  BLEU-{n_gram} without N400 features  :  {bleu_without_n400:.4f}")
    print(f"  BLEU-{n_gram} with    N400 features  :  {bleu_with_n400:.4f}")
    print(f"  Δ BLEU                            :  {delta:+.4f}")
    print(f"  Relative change                   :  {rel_change:+.1f}%")
    print()

    if delta > 0.001:
        verdict = "HELPED"
        detail  = "N400 features improved decoding performance."
    elif delta < -0.001:
        verdict = "HURT"
        detail  = (
            "N400 features reduced decoding performance.  "
            "Possible causes: overfitting (small dataset), noisy features "
            "(random EEG → random N400), or insufficient training epochs."
        )
    else:
        verdict = "NO EFFECT"
        detail  = "N400 features had negligible effect on decoding performance."

    print(f"  Verdict: N400 features {verdict}")
    print(f"  {detail}")
    print()
    print("  IMPORTANT NOTE ON FAKE DATA:")
    print("  TestEeg2024 is random Gaussian noise, so N400 = random scalars.")
    print("  Any performance difference is due to random variation, not real")
    print("  semantic signal.  Re-run this evaluation with real EEG data to")
    print("  obtain meaningful results.")
    print(sep)

    # Optional sanity check
    if words is not None and n400_values is not None:
        compute_n400_word_freq_correlation(words, n400_values)

    return {
        "bleu_with":          bleu_with_n400,
        "bleu_without":       bleu_without_n400,
        "delta":              delta,
        "relative_change_pct": rel_change,
        "helped":             delta > 0,
    }


# ===========================================================================
# CONVENIENCE WRAPPER — used by BrainModule during forward pass
# ===========================================================================

def compute_n400_from_batch(
    eeg_batch: np.ndarray,
    channel_names: Optional[List[str]] = None,
    sfreq: float = 50.0,
) -> np.ndarray:
    """
    Extract N400 and theta features from a single mini-batch of EEG epochs.

    This is the lightweight wrapper called inside the model's forward pass.
    It returns a (batch_size, 2) float32 array ready for concatenation with
    the brain embedding tensor.

    Column 0: z-scored N400 amplitude
    Column 1: z-scored theta power

    Parameters
    ----------
    eeg_batch : np.ndarray, shape (batch_size, n_channels, n_timepoints)
    channel_names : list of str or None
    sfreq : float

    Returns
    -------
    np.ndarray, shape (batch_size, 2), dtype float32

    Performance note
    ----------------
    Computing N400 on every forward pass adds a small CPU overhead (~1 ms
    for a 32-sample batch at 50 Hz).  For large datasets or real-time
    inference, pre-compute and cache these features alongside the EEG epochs.
    """
    feats = extract_n400_features(eeg_batch, channel_names=channel_names, sfreq=sfreq)

    # Stack into (batch_size, 2): column 0 = N400, column 1 = theta
    combined = np.stack([feats["n400"], feats["theta"]], axis=1).astype(np.float32)

    # Replace any NaN that might arise from degenerate batches
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    return combined
