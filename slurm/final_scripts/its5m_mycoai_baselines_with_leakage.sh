#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) MycoAI-BERT / MycoAI-CNN baseline comparison, WITH
# leakage included -- counterpart to its5m_mycoai_baselines.sh. Same
# checkpoints, same KNN pipeline (knn_its_mycoai.py, unchanged), only
# --tasks-dir differs (points at tasks_with_leakage/ instead of tasks/).
#
# REQUIRES:
#   - its_export_tasks_with_leakage.sh already run
#   - MycoAI-BERT.pt and MycoAI-CNN.pt already downloaded (same checkpoints
#     as its5m_mycoai_baselines.sh)
#
# Results: results_final/KNN_ITS_external_with_leakage_RESULTS.txt -- same
# file the other with-leakage ITS baselines write to.
#
# Submit: sbatch slurm/final_scripts/its5m_mycoai_baselines_with_leakage.sh
# ============================================================================
#SBATCH --job-name=its5m_mycoai_with_leakage
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/project/6045013/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks_with_leakage"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks_with_leakage.sh first" && exit 1
CHECKPOINT_DIR="/scratch/$USER/mycoai_models"
TEMPERATURE=0.07

# ── Grid (2 tasks) ──────────────────────────────────────────────────────────
CHECKPOINTS=(
    "${CHECKPOINT_DIR}/MycoAI-BERT.pt"
    "${CHECKPOINT_DIR}/MycoAI-CNN.pt"
)
TAGS=("mycoai_bert" "mycoai_cnn")

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
[ ! -f "${CHECKPOINT}" ] && echo "ERROR: ${CHECKPOINT} not found" && exit 1

echo "Checkpoint: ${CHECKPOINT} | tag: ${TAG}"

OVERALL_EXIT=0
for WEIGHTS in "uniform" "softmax"; do
    WEIGHT_ARGS=(--knn-weights "${WEIGHTS}")
    [ "${WEIGHTS}" = "softmax" ] && WEIGHT_ARGS+=(--temperature ${TEMPERATURE})
    python barcodebert/knn_its_mycoai.py \
        --checkpoint               "${CHECKPOINT}" \
        --data-dir                 "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --n-neighbors               1 3 5 7 10 15 20 25 50 \
        --metric                    cosine \
        "${WEIGHT_ARGS[@]}" \
        --run-name                  "knn_external_${TAG}_${WEIGHTS}_with_leakage" \
        --results-file              results_final/KNN_ITS_external_with_leakage_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: knn_its_mycoai.py failed for ${TAG}/${WEIGHTS}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}