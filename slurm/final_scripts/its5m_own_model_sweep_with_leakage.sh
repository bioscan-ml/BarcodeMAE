#!/bin/bash
# ============================================================================
# ITS-5M: uniform KNN + softmax-KNN temperature sweep for our own final
# BarcodeMAE+ configuration, WITH leakage included -- counterpart to
# its5m_own_model_sweep.sh. Same checkpoint (+CLS+Binary, CLS
# representation), same KNN pipeline (knn_its_clean.py, unchanged), only
# --tasks-dir differs (points at tasks_with_leakage/ instead of tasks/).
#
# REQUIRES its_export_tasks_with_leakage.sh to have been run first.
#
# Results: results_final/KNN_ITS_own_model_sweep_with_leakage_RESULTS.txt
#
# Submit: sbatch slurm/final_scripts/its5m_own_model_sweep_with_leakage.sh
# ============================================================================
#SBATCH --job-name=its5m_own_model_sweep_with_leakage
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=final_logs/%j/%j.out
#SBATCH --error=final_logs/%j/%j.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/job_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_JOB_ID}"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks_with_leakage"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks_with_leakage.sh first" && exit 1

MAIN_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/${DATASET}"
CKPT="${MAIN_CKPT_ROOT}/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

echo "=== UNIFORM KNN EVALUATION ==="
python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type cls \
    --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine \
    --knn-weights uniform \
    --run-name knn_own_model_uniform_with_leakage \
    --results-file results_final/KNN_ITS_own_model_sweep_with_leakage_RESULTS.txt \
    --log-wandb --wandb-project barcodemae_cls
EC0=$?; [ ${EC0} -ne 0 ] && echo "ERROR: uniform KNN eval failed"

echo "=== SOFTMAX TEMPERATURE SWEEP ==="
python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type cls \
    --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine \
    --knn-weights softmax --temperature-sweep ${TEMPS} \
    --run-name knn_own_model_softmax_sweep_with_leakage \
    --results-file results_final/KNN_ITS_own_model_sweep_with_leakage_RESULTS.txt \
    --log-wandb --wandb-project barcodemae_cls
EC1=$?; [ ${EC1} -ne 0 ] && echo "ERROR: temperature sweep failed"

OVERALL_EXIT=0
[ ${EC0} -ne 0 ] && OVERALL_EXIT=${EC0}
[ ${EC1} -ne 0 ] && OVERALL_EXIT=${EC1}
echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}