#!/bin/bash
# ============================================================================
# BIOSCAN-5M: zero-shot BIN reconstruction (ZSC) for HyenaDNA-tiny only.
# HyenaDNA-tiny's row in tab:bioscan_external currently has no BIN AMI/HM
# ("--" for both) -- this fills those two cells in.
#
# REQUIRES slurm/setup_env_modern.sh to have been run once first.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_hyenadna_zsc.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_hyenadna_zsc
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=final_logs/%j/%j.out
#SBATCH --error=final_logs/%j/%j.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv_modern/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
# mycoai's __init__.py calls wandb.login('allow') at import time regardless
# of WANDB_MODE, which starts a local service subprocess that writes a port
# file under $TMPDIR -- this cluster's node-local /tmp intermittently fails
# that write (confirmed repeatedly this session). Point TMPDIR at scratch
# instead, which doesn't have that flakiness.
export TMPDIR="/scratch/$USER/tmp_wandb"
mkdir -p "$TMPDIR"

mkdir -p results_final
mkdir -p "final_logs/${SLURM_JOB_ID}"

DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
MODEL_ID="LongSafari/hyenadna-tiny-1k-seqlen-hf"
MODEL_CLS="causal-lm"

echo "=== ZSC EVALUATION: HyenaDNA-tiny ==="
python barcodebert/zsc_evaluation_v2.py \
    --external-model-id     "${MODEL_ID}" \
    --external-model-cls    "${MODEL_CLS}" \
    --external-max-length   660 \
    --dataset                "${DATASET}" \
    --data-dir                "${DATA_DIR}" \
    --taxon                   genus \
    --n-neighbors              15 \
    --metric                   cosine \
    --run-name                  "zsc_external_hyenadna_tiny" \
    --results-file               results_final/ZSC_external_RESULTS.txt
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}