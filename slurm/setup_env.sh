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

# ── 3. pip / wheel ───────────────────────────────────────────────────────────
pip install --upgrade pip wheel --quiet

# ── 4. PyTorch family — all from the same cu121 index to avoid binary mismatches
echo "Installing PyTorch 2.1.1 + torchtext/torchdata (cu121 wheels) ..."
pip install \
    torch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    torchtext==0.16.1 \
    torchdata==0.7.1 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 5. pkg_resources stub (cluster setuptools omits it; wandb==0.15.12 needs it)
echo "Creating pkg_resources stub ..."
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/pkg_resources"
cat > "$SITE/pkg_resources/__init__.py" << 'EOF'
# Minimal stub so wandb==0.15.12 can import pkg_resources on this cluster.
from importlib.metadata import version as _version, packages_distributions as _pd
def get_distribution(name):
    class _Dist:
        def __init__(self, n):
            self.version = _version(n)
    return _Dist(name)
EOF

# ── 6. requirements_lambda.txt (skip torch-family lines already installed) ───
echo "Installing requirements_lambda.txt ..."
grep -v -E "^(torch|#|$)" "$REPO_DIR/requirements_lambda.txt" | \
    pip install -r /dev/stdin --quiet

# ── 7. barcodebert editable install ──────────────────────────────────────────
echo "Installing barcodebert (editable) ..."
pip install -e "$REPO_DIR" --no-deps --quiet

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Smoke test"
echo "=========================================="
python -c "
import torch
print(f'torch        : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')

import torchtext
print(f'torchtext    : {torchtext.__version__}')

import transformers
print(f'transformers : {transformers.__version__}')

import pkg_resources
print(f'pkg_resources: OK')

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