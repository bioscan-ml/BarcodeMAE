#!/bin/bash
# ============================================================================
# Third, isolated environment for evaluating BarcodeMamba/BarcodeMamba+
# checkpoints (https://github.com/bioscan-ml/BarcodeMamba-dev, branch
# GTCtech-BarcodeMambaPlus-release) via knn_probing_barcodemamba.py /
# knn_its_barcodemamba.py.
#
# Why a THIRD venv: BarcodeMamba's own pyproject.toml pins torch==2.3, with
# its own specific mamba_ssm/causal-conv1d wheel builds (cu12torch2.3) --
# different from both our main venv (torch 2.1.1) and modern venv (torch
# 2.5.1, its own different mamba_ssm build for Caduceus). Mixing these in one
# venv is not possible (compiled-extension ABI is tied to the exact torch
# build), so BarcodeMamba gets its own.
#
# This ALSO clones the BarcodeMamba-dev repo itself (for utils.barcode_mamba
# .BarcodeMamba, needed regardless of tokenizer type) and installs
# mycoai-its (needed only to unpickle bpe_tokenizer.pkl -- see
# barcodebert/barcodemamba_common.py's docstring for why the repo's own
# vendored utils/mycoai/ copy can't be used for this instead).
#
# Run once from the BarcodeMAE repo root, on the LOGIN node (needs internet
# for the repo clone + mamba_ssm/causal-conv1d GitHub release wheels):
#   bash slurm/setup_env_barcodemamba.sh
#
# After this completes, SLURM jobs activate it with:
#   source "/scratch/$USER/BarcodeMAE_venv_barcodemamba/bin/activate"
# and the cloned repo lives at:
#   /scratch/$USER/BarcodeMamba-dev

set -e

echo "=========================================="
echo "BarcodeMamba evaluation environment setup"
echo "User   : $USER"
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Load cluster modules ───────────────────────────────────────────────
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

echo "Python : $(which python3)"

# ── 2. Clone BarcodeMamba-dev (branch GTCtech-BarcodeMambaPlus-release) ───
# NOTE: this repo is access-restricted, and narval has no GitHub credentials
# to clone/pull it directly. If $BM_REPO already exists (e.g. rsynced over
# from a local clone made where you DO have GitHub access), we use it as-is
# and skip fetch/pull -- do not delete/re-clone it here.
BM_REPO="/scratch/$USER/BarcodeMamba-dev"
if [ -d "$BM_REPO" ]; then
    echo "BarcodeMamba-dev already exists at $BM_REPO — using as-is (skipping fetch/pull, no GitHub credentials on this host)."
else
    git clone --branch GTCtech-BarcodeMambaPlus-release --single-branch \
        https://github.com/bioscan-ml/BarcodeMamba-dev.git "$BM_REPO"
fi

# ── 3. Create virtual environment ──────────────────────────────────────────
VENV_PATH="/scratch/$USER/BarcodeMAE_venv_barcodemamba"

if [ -d "$VENV_PATH" ]; then
    echo "Venv exists at $VENV_PATH — delete it to rebuild: rm -rf $VENV_PATH"
else
    echo "Creating venv at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
echo "Activated: $(which python)"

# ── 4. pip / wheel ─────────────────────────────────────────────────────────
pip install --upgrade pip wheel --quiet

# ── 5. PyTorch 2.3 (BarcodeMamba's pinned version) ─────────────────────────
echo "Installing PyTorch 2.3 ..."
pip install "torch==2.3.1+computecanada" --quiet || pip install "torch==2.3.1" --quiet

# ── 6. mamba_ssm / causal-conv1d -- cu12, torch2.3, cp311, cxx11abiTRUE.
#      NOTE: cxx11abiTRUE (not FALSE) -- PyPI's torch==2.3.1 is built with the
#      new (cxx11) C++ ABI, so the cxx11abiFALSE wheels fail to import with an
#      "undefined symbol" error from selective_scan_cuda.so (mismatched
#      std::string/c10::Warning symbol mangling between the two ABIs). ──────
echo "Installing mamba_ssm + causal-conv1d (cxx11abiTRUE, matching PyPI torch's ABI) ..."
pip install --quiet \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.3cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
pip install --quiet \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.3cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# ── 7. pkg_resources stub (cluster setuptools omits it; wandb==0.18 needs it) ─
echo "Creating pkg_resources stub ..."
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/pkg_resources"
cat > "$SITE/pkg_resources/__init__.py" << 'EOF'
from importlib.metadata import version as _version
from packaging.version import parse as parse_version  # noqa: F401 (re-exported)


def get_distribution(name):
    class _Dist:
        def __init__(self, n):
            self.version = _version(n)

    return _Dist(name)
EOF

# ── 8. Remaining pyproject.toml dependencies (torch/mamba_ssm/causal-conv1d
#      already installed above; torchtext is fine here since torch==2.3 has a
#      matching build, unlike the modern venv's torch 2.5.1) ───────────────
echo "Installing remaining BarcodeMamba dependencies ..."
pip install --quiet \
    "biopython>=1.86" "bioscan-dataset>=1.3.0" "boto3>=1.42.34" \
    "einops>=0.8.1" "hydra-core>=1.3.2" "lightning>=2.6.0" \
    "matplotlib>=3.10.8" "pandas>=2.3.3" "plotly>=6.5.2" "rich>=14.3.1" \
    "scikit-learn>=1.8.0" "sentencepiece>=0.2.1" "tensorboard>=2.20.0" \
    "timm>=1.0.24" "torchmetrics>=1.8.2" "torchtext>=0.18.0" "tqdm>=4.67.1" \
    "transformers>=4.42.3,<5.0" "triton>=2.3.0" "wandb==0.18"

# ── 9. mycoai-its -- needed only to unpickle bpe_tokenizer.pkl (see
#      barcodemamba_common.py docstring) ────────────────────────────────────
echo "Installing mycoai-its ..."
pip install "mycoai-its==0.0.5" --index-url https://pypi.org/simple/ --quiet

# ── 10. barcodebert editable install ────────────────────────────────────────
echo "Installing barcodebert (editable) ..."
pip install -e "$REPO_DIR" --no-deps --quiet

# ── 11. Smoke test ───────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Smoke test"
echo "=========================================="
# mycoai's __init__.py calls wandb.login('allow') at import time regardless
# of WANDB_MODE, starting a local service subprocess that writes a port file
# under $TMPDIR -- this cluster's node-local /tmp intermittently fails that
# write. Scratch doesn't have that flakiness.
export TMPDIR="/scratch/$USER/tmp_wandb"
mkdir -p "$TMPDIR"
WANDB_MODE=disabled python -c "
import torch
print(f'torch        : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')

import mamba_ssm
print(f'mamba_ssm    : OK')

import causal_conv1d
print(f'causal_conv1d: OK')

import hydra
from omegaconf import OmegaConf
print(f'hydra/omegaconf: OK')

import lightning
print(f'lightning    : OK')

import mycoai
from mycoai.data.encoders import BytePairEncoder
print(f'mycoai       : OK (BytePairEncoder importable)')

import sys
sys.path.insert(0, '$BM_REPO')
from utils.barcode_mamba import BarcodeMamba
print(f'BarcodeMamba : OK (from $BM_REPO)')

import barcodebert
from barcodebert.barcodemamba_common import load_barcodemamba, load_bpe_tokenizer, embed_sequences
print(f'barcodebert.barcodemamba_common: OK')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with:  source $VENV_PATH/bin/activate"
echo "Repo cloned at: $BM_REPO"
echo "=========================================="