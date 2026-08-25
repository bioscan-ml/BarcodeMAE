#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set genus-level KNN eval for BIOSCAN-5M CE
# aux-weight ablation checkpoints (0.01/0.05/0.50/1.00) plus the w=0.10
# main-sweep checkpoint, on the cluster whose repo root is
# /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE (same cluster as
# its_binary_val_eval.sh). Only ce is included -- binary/triplet ablation
# checkpoints on this cluster were still at epoch 5-6/35 as of the last
# MLM-accuracy check, not ready for eval yet.
#
# One array task per weight (5 total: 0.01/0.05/0.50/1.00 + 0.10 main).
# BIOSCAN-5M's gallery (~120K specimens) is much smaller than ITS-5M's
# (~5.2M), so this should run well within the hour, but the walltime is set
# generously anyway.
#
# *** GPU TYPE: h100, matching the original main-sweep scripts for this
# cluster. If wrong, `sbatch --test-only` will say so immediately -- same
# check that caught narval's h100->a100 issue.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_ce_val_eval.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_ce_val_eval
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
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

ABL_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/BIOSCAN-5M"
MAIN_CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/BIOSCAN-5M/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
RESULTS_FILE="results_final/KNN_val_bioscan5m_aux_weight_ablation_RESULTS.txt"

WEIGHTS=(0.01 0.05 0.50 1.00 0.10)
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
echo "ce w=${WEIGHT}"

if [ "${WEIGHT}" = "0.10" ]; then
    CKPT="${MAIN_CKPT}"
    RUN_NAME="val_final_bioscan5m_ce_w0.10"
else
    CKPT="${ABL_BASE}/ablw_bioscan5m_k6_6L6H_6DL6DH_maelm_cls_ce_w${WEIGHT}/checkpoint_encoder.pt"
    RUN_NAME="val_ablw_bioscan5m_ce_w${WEIGHT}"
fi
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

python barcodebert/knn_probing.py \
    --pretrained-checkpoint "${CKPT}" \
    --dataset BIOSCAN-5M --data-dir "${DATA_DIR}" --query-file supervised_val.csv \
    --representation_type cls --taxon genus \
    --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
    --run-name "${RUN_NAME}" \
    --results-file "${RESULTS_FILE}"
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
