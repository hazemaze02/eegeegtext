"""Test script using the Broderick2019 EEG dataset.

Modified version of test.py that:
- Uses Broderick2019 (an EEG dataset, CC0 1.0 licence, auto-downloadable from Dryad)
- Downloads data automatically on first run
- Runs with a single subject/timeline for a quick smoke test
- Disables Weights & Biases logging (no account needed)
- Uses t5-small instead of t5-large for faster first run

Run from the REPO ROOT directory:
    python -m sentence_decoding.grids.test_eeg

Environment variables (optional — defaults shown below):
    DATAPATH   where raw data is stored  (default: ./data)
    SAVEPATH   where results are saved   (default: ./results)
"""

import os
import sys

# ------------------------------------------------------------------ #
# Resolve paths BEFORE importing defaults (defaults.py reads env vars
# at import time, so we must set them first).
# ------------------------------------------------------------------ #
if not os.environ.get("DATAPATH"):
    os.environ["DATAPATH"] = os.path.join(os.getcwd(), "data")
if not os.environ.get("SAVEPATH"):
    # SAVEPATH is the BASE directory — the code appends "results/" and "cache/" itself
    os.environ["SAVEPATH"] = os.getcwd()

DATA_PATH = os.environ["DATAPATH"]
SAVE_PATH = os.environ["SAVEPATH"]
os.makedirs(DATA_PATH, exist_ok=True)
# Create cache and results dirs upfront — pydantic validates these exist before training
os.makedirs(os.path.join(SAVE_PATH, "cache", "sentence_decoding"), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "results", "sentence_decoding"), exist_ok=True)

print(f"[setup] DATAPATH : {DATA_PATH}")
print(f"[setup] SAVEPATH : {SAVE_PATH}")

# ------------------------------------------------------------------ #
# Now import the rest
# ------------------------------------------------------------------ #
import neuralset.studies.testeeg2024  # noqa: F401 — registers TestEeg2024
from sentence_decoding.main import Experiment as Exp       # noqa: E402
from neuraltrain.utils import update_config                 # noqa: E402
from .defaults import default_config                        # noqa: E402


# ------------------------------------------------------------------ #
# Download the dataset if not present yet
# ------------------------------------------------------------------ #
def download_broderick_if_needed(data_path: str) -> None:
    """Download Broderick2019 from Dryad if the folder is missing/empty."""
    import neuralset as ns
    from neuralset.data import _get_study

    study_cls = _get_study("Broderick2019")
    study_path = os.path.join(data_path, "Broderick2019")

    if os.path.isdir(study_path) and any(os.scandir(study_path)):
        print(f"[data] Broderick2019 found at {study_path}, skipping download.")
        return

    print(
        "[data] Broderick2019 not found — downloading from Dryad (~3-5 GB).\n"
        "       This may take a while depending on your connection speed...\n"
        "       Note: a 'private files' download step may fail (these are not\n"
        "       needed for the Natural Speech / word-decoding task).\n"
    )
    os.makedirs(study_path, exist_ok=True)
    try:
        study_cls.download(study_path)
    except Exception as e:
        print(f"[data] Download warning (possibly only private files failed): {e}")
        print("[data] Continuing — the public Natural Speech data should still work.")


# ------------------------------------------------------------------ #
# Experiment config overrides
# ------------------------------------------------------------------ #
default_params = {
    # Run locally (no Slurm)
    "infra.cluster": None,
    # Remove 'projects' from workdir — that folder doesn't exist locally
    "infra.workdir": {"copied": ["neuralset", "neuraltrain"], "includes": ["*.py"]},
    # Load only 1 recording timeline to keep things fast
    "data.n_timelines": 1,
    # Skip saving checkpoints during quick testing
    "save_checkpoints": False,
    # Disable W&B (no account needed)
    "use_wandb": False,
    # Very short training for smoke-test purposes
    "trainer_config.n_epochs": 10,
    "trainer_config.patience": 10,
    # Fewer workers — safer on a laptop (increase if you have more cores)
    "data.num_workers": 2,
    # Use the smaller t5-small instead of t5-large (much faster, still valid)
    "data.feature.model_name": "t5-small",
    # Smaller batch size for EEG
    "data.batch_size": 32,
}

default = update_config(default_config, default_params)

# TestEeg2024 is our synthetic EEG dataset — no download needed, data is
# generated on the fly. main.py recognises it as EEG automatically.
params = {"data.dataset": "TestEeg2024"}


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # TestEeg2024 generates synthetic data on the fly — no download needed

    # Build final config
    config = update_config(default, params)

    # Patch paths now that env vars are live
    from sentence_decoding.grids.defaults import CACHEDIR, SAVEDIR  # noqa: E402
    config["cache"] = CACHEDIR
    config["data"]["cache"] = CACHEDIR
    config["data"]["data_path"] = DATA_PATH
    config["infra"]["folder"] = SAVEDIR

    # Give this test run its own subfolder
    job_name = "|".join([f"{k}={v}" for k, v in params.items()])
    run_folder = os.path.join(SAVEDIR, "test_eeg", job_name)
    config["infra"]["folder"] = run_folder

    # Clear any previous test run
    if os.path.exists(run_folder):
        import shutil
        shutil.rmtree(run_folder)
    os.makedirs(run_folder, exist_ok=True)

    print(f"\n[run] Starting experiment. Results → {run_folder}\n")

    task = Exp(**config)
    task.infra.clear_job()
    out = task.run()

    print("\n=== Finished! Check your results folder for outputs. ===")
    print(f"    {run_folder}")
