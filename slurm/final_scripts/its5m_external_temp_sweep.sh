#!/bin/bash
# ============================================================================
# ITS-5M external baselines: softmax-KNN temperature sweep, for the 4 models
# that work in the main BarcodeMAE_venv. Uniform-KNN results already exist
# (KNN_ITS_external_RESULTS.txt) -- this only adds the new softmax column,
# sweeping T and reporting the best (T, k) combo per model per test set.
#
# See its5m_external_temp_sweep_modern.sh for the 3 models needing the
# modern venv (HyenaDNA-tiny, Caduceus-PS-1k, GENA-LM).
#
# Submit: sbatch slurm/final_scripts/its5m_external_temp_sweep.sh
# ============================================================================
#SBATCH --job-name=its5m_external_temp_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

MODEL_IDS=(
    "zhihan1996/DNABERT-2-117M"
    "zhihan1996/DNABERT-S"
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    "bioscan-ml/BarcodeBERT"
)
MODEL_CLS=(
    "auto" "auto" "masked-lm" "auto"
)
MODEL_TAGS=(
    "dnabert2" "dnaberts" "nucleotide_transformer" "barcodebert"
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