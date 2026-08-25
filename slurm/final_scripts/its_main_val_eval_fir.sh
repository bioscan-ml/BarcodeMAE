#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set genus-level KNN eval for the ITS-5M w=0.10
# main-sweep checkpoints (binary, ce, triplet), on the "fir" cluster --
# repo root /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE
# (same relative path as narval's checkout, but a different cluster/home --
# confirmed this is where the ITS-5M main-sweep checkpoints actually live).
#
# Only ITS-5M/binary's w=0.10 (task 0) is decision-critical (the weight
# being chosen for the paper); ce and triplet (tasks 1-2) are included for
# the complete-picture comparison, same as aux_weight_val_eval_othertasks.sh
# on narval.
#
# REQUIRES data/ITS-5M/tasks/trainset_valid_tasks.csv already exported on
# THIS cluster (via analyze_its_valtrain_overlap.py) -- check it exists
# before submitting; if not, export it first (see that script's usage).
#
# GPU TYPE: h100 (confirmed for this cluster).
#
# Submit: sbatch slurm/final_scripts/its_main_val_eval_fir.sh
# ============================================================================
#SBATCH --job-name=its_main_val_eval_fir
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"
export WANDB_MODE=disabled

mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

REPO_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE"
DATA_DIR="${REPO_ROOT}/data/ITS-5M"
MAIN_BASE="${REPO_ROOT}/main_checkpoints_final/ITS-5M"

AUX_TASKS=("binary" "ce" "triplet")
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "aux_task: ${AUX_TASK} (w=0.10, main)"

CKPT="${MAIN_BASE}/final_its_k6_6L6H_6DL6DH_maelm_cls_${AUX_TASK}/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

python barcodebert/knn_its_clean_val.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${DATA_DIR}/tasks" \
    --representation-type cls \
    --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
    --run-name "val_final_its_${AUX_TASK}_w0.10" \
    --results-file results_final/KNN_val_ITS_aux_weight_ablation_RESULTS.txt
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}