#!/bin/bash
# ============================================================================
# ITS-5M external baselines: softmax-KNN temperature sweep, for the 3 models
# that need the modern venv -- HyenaDNA-tiny, Caduceus-PS-1k, GENA-LM. See
# its5m_external_temp_sweep.sh for the other 4 (main-venv) models.
#
# REQUIRES slurm/setup_env_modern.sh to have been run once first.
#
# Submit: sbatch slurm/final_scripts/its5m_external_temp_sweep_modern.sh
# ============================================================================
#SBATCH --job-name=its5m_external_temp_sweep_modern
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

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
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

MODEL_IDS=(
    "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
    "AIRI-Institute/moderngena-base"
)
MODEL_CLS=(
    "causal-lm" "masked-lm" "auto"
)
MODEL_TAGS=(
    "hyenadna_tiny" "caduceus_ps1k" "gena_lm"
)

MODEL_ID="${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}"
MODEL_CLS_ARG="${MODEL_CLS[$SLURM_ARRAY_TASK_ID]}"
TAG="${MODEL_TAGS[$SLURM_ARRAY_TASK_ID]}"
echo "Model: ${MODEL_ID} | class: ${MODEL_CLS_ARG} | tag: ${TAG}"

python barcodebert/knn_its_clean.py \
    --external-model-id     "${MODEL_ID}" \
    --external-model-cls    "${MODEL_CLS_ARG}" \
    --external-max-length   660 \
    --data-dir                "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type     tokens \
    --embed-batch-size        32 \
    --n-neighbors              1 3 5 7 10 15 20 25 50 \
    --metric                   cosine \
    --knn-weights              softmax \
    --temperature-sweep        ${TEMPS} \
    --run-name                  "knn_its_external_${TAG}_softmax_sweep" \
    --results-file               results_final/KNN_ITS_external_temp_sweep_RESULTS.txt
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}