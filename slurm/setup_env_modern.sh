#!/bin/bash
# Second, isolated environment for external baseline models that need a newer
# torch/transformers than the main BarcodeMAE_venv supports.
#
# Why this exists: Caduceus's remote code hard-requires mamba_ssm, which
# requires torch~=2.5.0. But torch~=2.5.0 has no matching torchtext build on
# this cluster's wheelhouse (torchtext tops out around torch 2.3). GENA-LM
# (ModernBERT architecture) and HyenaDNA also need a newer `transformers`
# than the main venv's pinned 4.29.2 to load via AutoConfig/AutoModel.
# torchtext itself is no longer a hard blocker (barcodebert's eval scripts
# were patched to import it lazily, only when building the *internal*
# k-mer vocab -- external model evaluation never touches it), so this venv
# skips torchtext entirely and is free to use a modern torch.
#
# Run once from the BarcodeMAE repo root:
#   bash slurm/setup_env_modern.sh
#
# After this completes, SLURM jobs activate it with:
#   source "/scratch/$USER/BarcodeMAE_venv_modern/bin/activate"

set -e

echo "=========================================="
echo "BarcodeMAE modern-stack environment setup"
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
VENV_PATH="/scratch/$USER/BarcodeMAE_venv_modern"

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

# ── 4. PyTorch -- torch only, no torchvision/torchaudio/torchtext needed for
#      external-model embedding extraction, and none of the wheelhouse's
#      torchtext builds support torch 2.5.x anyway.
echo "Installing PyTorch 2.5.1 (matches mamba_ssm's torch~=2.5.0 requirement) ..."
pip install torch==2.5.1+computecanada --quiet

# ── 5. mamba_ssm / causal-conv1d -- required (hard import, not optional) by
#      Caduceus's remote modeling code.
echo "Installing mamba_ssm + causal-conv1d ..."
pip install causal-conv1d==1.5.0.post8+computecanada --quiet
pip install mamba-ssm==2.2.4+computecanada --quiet

# ── 6. pkg_resources stub (cluster setuptools omits it; wandb==0.15.12 needs it)
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

# ── 7. requirements_lambda.txt (skip torch-family/transformers/tokenizers --
#      the file pins tokenizers==0.13.3 for the OLD transformers==4.29.2 the
#      main venv uses; installing that here with normal dependency resolution
#      would silently drag transformers back down to stay compatible with it) ─
echo "Installing requirements_lambda.txt (excluding torch-family/transformers/tokenizers) ..."
grep -v -E "^(torch|transformers|tokenizers|#|$)" "$REPO_DIR/requirements_lambda.txt" | \
    pip install -r /dev/stdin --quiet

# ── 7b. mycoai-its -- install explicitly from PyPI in case cluster wheelhouse skips it
echo "Installing mycoai-its ..."
pip install mycoai-its==0.0.5 --index-url https://pypi.org/simple/ --quiet

# ── 8. transformers -- installed LAST (after everything else that might pull
#      in an incompatible tokenizers/transformers pin), with normal dependency
#      resolution so it can pull in whatever tokenizers version it actually
#      needs. Oldest version that supports ModernBERT (GENA-LM) and has
#      correct AutoConfig/trust_remote_code fallback behaviour (HyenaDNA) --
#      deliberately not jumping to latest, to minimize the chance of breaking
#      the already-working DNABERT-2/DNABERT-S/NT/BarcodeBERT loading code.
echo "Installing transformers 4.48.0 (ModernBERT support) ..."
pip install --force-reinstall transformers==4.48.0+computecanada --quiet

# ── 8b. Re-pin numpy<2 -- one of the installs above (transformers 4.48.0's
#      own dependency resolution) silently upgrades numpy to 2.x with no
#      warning at install time, but scipy/scikit-learn/numba/mamba_ssm's
#      compiled C extensions here were built against NumPy 1.x's ABI, which
#      NumPy 2.0 broke ("_ARRAY_API not found" / "numpy.core.multiarray
#      failed to import" the moment mamba_ssm pulls in sklearn transitively
#      via transformers.generation). Confirmed via a real cluster traceback.
echo "Re-pinning numpy<2 (scipy/sklearn/mamba_ssm need NumPy 1.x ABI) ..."
pip install "numpy<2" --quiet
python -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__; print(f'numpy: {numpy.__version__} OK')"

# ── 9. barcodebert editable install ──────────────────────────────────────────
echo "Installing barcodebert (editable) ..."
pip install -e "$REPO_DIR" --no-deps --quiet

# ── 10. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Smoke test"
echo "=========================================="
WANDB_MODE=disabled python -c "
import numpy
assert numpy.__version__.startswith('1.'), f'numpy is {numpy.__version__}, expected 1.x'
print(f'numpy        : {numpy.__version__}')

import torch
print(f'torch        : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')

import mamba_ssm
print(f'mamba_ssm    : OK')

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

from barcodebert.external_models import load_external_model
print(f'external_models: OK')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with:  source $VENV_PATH/bin/activate"
echo "=========================================="