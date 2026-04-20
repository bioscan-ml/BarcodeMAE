#!/bin/bash
# One-shot environment setup for BarcodeMAE on a new cluster.
# Run once from the BarcodeMAE repo root:
#   bash slurm/setup_env.sh
#
# After this completes, all SLURM jobs activate the venv with:
#   source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

set -e  # exit immediately on any error

echo "=========================================="
echo "BarcodeMAE environment setup"
echo "User   : $USER"
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "=========================================="

# ── 1. Load cluster modules ───────────────────────────────────────────────────
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

echo "Python : $(which python3)"
echo "CUDA   : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"

# ── 2. Create virtual environment in scratch (fast NVMe storage) ──────────────
VENV_PATH="/scratch/$USER/BarcodeMAE_venv"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    echo "To rebuild it from scratch, delete it first: rm -rf $VENV_PATH"
else
    echo "Creating virtual environment at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
echo "Activated venv: $(which python)"

# ── 3. Upgrade pip/wheel ──────────────────────────────────────────────────────
pip install --upgrade pip wheel setuptools --quiet

# ── 4. Install PyTorch with CUDA 12.1 wheels ─────────────────────────────────
# torch 2.1.1 + CUDA 12.1 is the closest stable wheel to CUDA 12.6
echo "Installing PyTorch 2.1.1 (CUDA 12.1 wheels) ..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 5. Install torchtext and torchdata (must match torch version) ─────────────
echo "Installing torchtext / torchdata ..."
pip install torchtext==0.16.1 torchdata==0.7.1 --quiet

# ── 6. Install remaining requirements ────────────────────────────────────────
echo "Installing requirements.txt ..."
# Install everything except torch-family packages already installed above
pip install \
    accelerate==0.25.0 \
    einops==0.6 \
    matplotlib \
    numpy==1.25.2 \
    omegaconf==2.3 \
    opt-einsum==3.3 \
    pandas==2.1 \
    peft==0.5 \
    scikit-learn==1.3 \
    scipy==1.12 \
    seaborn \
    transformers==4.29.2 \
    umap-learn==0.5.6 \
    wandb \
    --quiet

# ── 7. Install mycoai (required for ITS-5M dataset loading) ──────────────────
echo "Installing mycoai ..."
pip install mycoai --quiet

# ── 8. Install barcodebert package in editable mode ──────────────────────────
echo "Installing barcodebert package (editable) ..."
# Must be run from the BarcodeMAE directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
pip install -e "$REPO_DIR" --quiet

# ── 9. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Smoke test"
echo "=========================================="
python -c "
import torch, transformers, torchtext, pandas, numpy, sklearn, wandb
print(f'torch       : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print(f'GPU count   : {torch.cuda.device_count()}')
print(f'transformers: {transformers.__version__}')
print(f'torchtext   : {torchtext.__version__}')
print(f'pandas      : {pandas.__version__}')
print(f'numpy       : {numpy.__version__}')
import barcodebert
print(f'barcodebert : OK')
try:
    import mycoai
    print(f'mycoai      : OK')
except ImportError as e:
    print(f'mycoai      : MISSING — {e}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with:"
echo "  source $VENV_PATH/bin/activate"
echo "=========================================="