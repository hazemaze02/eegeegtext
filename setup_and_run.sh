#!/bin/bash
# ============================================================
# Setup & Run Script for Sentence Decoding (EEG version)
# ============================================================
# Usage: run this from inside your Code folder:
#   cd /path/to/Code
#   bash setup_and_run.sh

set -e
cd "$(dirname "$0")"

ENVNAME="eeg_decoding"
PYTHON_VERSION="3.11"

echo "======================================================"
echo " Sentence Decoding — EEG Setup Script"
echo "======================================================"
echo ""

# ------------------------------------------------------------------ #
# 0. Create a fresh conda env with Python 3.10
#    (neuralset requires Python >=3.10 and pandas >=2.2.2)
# ------------------------------------------------------------------ #
echo "=== [0/6] Creating conda environment '$ENVNAME' (Python $PYTHON_VERSION) ==="

if conda env list | grep -q "^$ENVNAME "; then
    echo "  Environment '$ENVNAME' already exists — recreating with Python $PYTHON_VERSION."
    conda env remove -n "$ENVNAME" -y
fi
conda create -n "$ENVNAME" python=$PYTHON_VERSION -y
echo "  Created '$ENVNAME' with Python $PYTHON_VERSION."

# Run everything inside the new environment
PYTHON="conda run -n $ENVNAME python"
PIP="conda run -n $ENVNAME pip"

echo ""

# ------------------------------------------------------------------ #
# 1. Install neuralset
# ------------------------------------------------------------------ #
echo "=== [1/6] Installing neuralset ==="
$PIP install --config-settings editable_mode=strict -e neuralset/.
echo ""

# ------------------------------------------------------------------ #
# 2. Install neuraltrain
# ------------------------------------------------------------------ #
echo "=== [2/6] Installing neuraltrain ==="
$PIP install -e neuraltrain/.
echo ""

# ------------------------------------------------------------------ #
# 3. Install sentence_decoding dependencies
# ------------------------------------------------------------------ #
echo "=== [3/6] Installing additional dependencies ==="
$PIP install "lightning>=2.0.8" "pydantic>=2.5.0" scipy torchvision "torchmetrics>=1.1.2" "wandb>=0.15.11" x_transformers
$PIP install "git+https://github.com/braindecode/braindecode@master"
# T5 language model for text embeddings
$PIP install "transformers>=4.20.0" sentencepiece
# Spacy + English model for text preprocessing
$PIP install "spacy>=3.5.4"
$PYTHON -m spacy download en_core_web_md
# gdown is needed for Broderick2019 download
$PIP install gdown
# kenlm is needed for optional beam-search decoding
$PIP install kenlm || echo "  kenlm install failed — beam-search decoder disabled (not needed for basic runs)"
# wandb (optional — logging is disabled by default in test_eeg.py)
$PIP install wandb
echo ""

# ------------------------------------------------------------------ #
# 4. Set environment variables
# ------------------------------------------------------------------ #
echo "=== [4/6] Setting paths ==="
export DATAPATH="$(pwd)/data"
export SAVEPATH="$(pwd)/results"
mkdir -p "$DATAPATH" "$SAVEPATH"
echo "  DATAPATH=$DATAPATH"
echo "  SAVEPATH=$SAVEPATH"
echo ""

# ------------------------------------------------------------------ #
# 5. Quick sanity check
# ------------------------------------------------------------------ #
echo "=== [5/6] Sanity check — importing neuralset ==="
$PYTHON -c "import neuralset; print('  neuralset OK:', neuralset.__file__)"
$PYTHON -c "import neuraltrain; print('  neuraltrain OK')"
echo ""

# ------------------------------------------------------------------ #
# 6. Run the EEG test
# ------------------------------------------------------------------ #
echo "=== [6/6] Running EEG test ==="
echo ""
echo "  Dataset : Broderick2019 (EEG, CC0 licence, auto-downloaded from Dryad)"
echo "  Epochs  : 3 (smoke test — increase trainer_config.n_epochs in test_eeg.py for real training)"
echo "  W&B     : disabled"
echo ""
echo "  NOTE: First run downloads ~3-5 GB. Please be patient!"
echo ""

DATAPATH="$DATAPATH" SAVEPATH="$SAVEPATH" \
    conda run -n "$ENVNAME" python -m sentence_decoding.grids.test_eeg

echo ""
echo "======================================================"
echo " All done! To run again (after setup):"
echo "   conda activate $ENVNAME"
echo "   export DATAPATH=$(pwd)/data"
echo "   export SAVEPATH=$(pwd)/results"
echo "   python -m sentence_decoding.grids.test_eeg"
echo "======================================================"
