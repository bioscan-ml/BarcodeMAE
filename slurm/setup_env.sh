#!/bin/bash
# One-shot environment setup for BarcodeMAE on a new cluster.
# Run once from the BarcodeMAE repo root:
#   bash slurm/setup_env.sh
#
# After this completes, all SLURM jobs activate the venv with:
#   source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

set -e

echo "=========================================="
echo "BarcodeMAE environment setup"
echo "User   : $USER"
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Load cluster modules ───────────────────────────────────────────────────
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

echo "Python : $(which python3)"

# ── 2. Create virtual environment ────────────────────────────────────────────
VENV_PATH="/scratch/$USER/BarcodeMAE_venv"

if [ -d "$VENV_PATH" ]; then
    echo "Venv exists at $VENV_PATH — delete it to rebuild: rm -rf $VENV_PATH"
else
    echo "Creating venv at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
echo "Activated: $(which python)"

# ── 3. pip / wheel / setuptools ────────────────────────────────────────────────
pip install --upgrade pip wheel --quiet
# Pin setuptools to <70 — versions >=70 dropped the bundled pkg_resources module
# that wandb==0.15.12 depends on (pkg_resources.parse_version specifically).
# Do this BEFORE any other install: requirements_lambda.txt / mycoai-its / the
# barcodebert editable install can all transitively pull in a newer setuptools
# and silently reinstall a broken pkg_resources over whatever's here, so this
# pin must both come first and actually stick -- a hand-written pkg_resources
# stub is not robust to that (confirmed: it gets clobbered on some clusters).
echo "Pinning setuptools<70 (for pkg_resources.parse_version) ..."
pip install --force-reinstall "setuptools>=65,<70" --quiet
python -c "import pkg_resources; from pkg_resources import parse_version" \
    && echo "pkg_resources OK" \
    || { echo "ERROR: pkg_resources still missing after setuptools install"; exit 1; }

# ── 4. PyTorch family — all from the same cu121 index to avoid binary mismatches
echo "Installing PyTorch 2.1.1 + torchtext/torchdata (cu121 wheels) ..."
pip install \
    torch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    torchtext==0.16.1 \
    torchdata==0.7.1 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 6. requirements_lambda.txt (skip torch-family lines already installed) ───
echo "Installing requirements_lambda.txt ..."
grep -v -E "^(torch|#|$)" "$REPO_DIR/requirements_lambda.txt" | \
    pip install -r /dev/stdin --quiet

# ── 6b. mycoai-its — install explicitly from PyPI in case cluster wheelhouse skips it
echo "Installing mycoai-its ..."
pip install mycoai-its==0.0.5 --index-url https://pypi.org/simple/ --quiet

# ── 7. barcodebert editable install ──────────────────────────────────────────
echo "Installing barcodebert (editable) ..."
pip install -e "$REPO_DIR" --no-deps --quiet

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Smoke test"
echo "=========================================="
WANDB_MODE=disabled python -c "
import torch
print(f'torch        : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')

import torchtext
print(f'torchtext    : {torchtext.__version__}')

import transformers
print(f'transformers : {transformers.__version__}')

import pkg_resources
from pkg_resources import parse_version
print(f'pkg_resources: OK (parse_version OK)')

import wandb
print(f'wandb        : {wandb.__version__}')

from mycoai.data import Data
print(f'mycoai       : OK')

import barcodebert
print(f'barcodebert  : OK')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with:  source $VENV_PATH/bin/activate"
echo "=========================================="