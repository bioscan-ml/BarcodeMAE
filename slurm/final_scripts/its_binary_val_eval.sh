#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set genus-level KNN eval for ITS-5M binary
# aux-weight ablation checkpoints (0.01/0.05/0.5/1.0), on the cluster whose
# repo root is /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE (not
# the narval BarcodeMAE_final/BarcodeMAE checkout).
#
# The w=0.10 main-sweep checkpoint does NOT live on this cluster -- it's on
# a THIRD cluster ("fir"), at .../BarcodeMAE_final/BarcodeMAE/
# main_checkpoints_final/ITS-5M/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/
# (confirmed). See its_main_val_eval_fir.sh for that one instead -- it can't
# be added here since /home isn't shared across clusters.
#
# One array task per weight, so a walltime-kill on one doesn't lose the
# others. Each task re-embeds the ~5.2M-specimen trainset.fasta gallery from
# scratch (embeddings aren't shared across different checkpoints), so this
# needs a real GPU allocation, not a login-node/short-default run.
#
# *** GPU TYPE: this is set to h100, matching the original main-sweep
# scripts for this cluster (bioscan5m_final.sh / fungi_its_final.sh). If
# this cluster doesn't actually have h100s, `sbatch --test-only` will say so
# immediately (same failure narval gave for h100 -- see
# bioscan5m_aux_weight_ablation_home.sh's history) -- swap to whatever GPU
# type it reports as valid.
#
# REQUIRES data/ITS-5M/tasks/trainset_valid_tasks.csv already exported (via
# analyze_its_valtrain_overlap.py) -- already done for this checkout.
#
# Submit: sbatch slurm/final_scripts/its_binary_val_eval.sh
# ============================================================================
#SBATCH --job-name=its_binary_val_eval
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
export WANDB_MODE=disabled

mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/ITS-5M"

WEIGHTS=(0.01 0.05 0.5 1.0)
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
echo "binary w=${WEIGHT}"

CKPT="${CKPT_BASE}/ablw_its_k6_6L6H_6DL6DH_maelm_cls_binary_w${WEIGHT}/checkpoint_encoder.pt"
RUN_NAME="val_ablw_its_binary_w${WEIGHT}"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

python barcodebert/knn_its_clean_val.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${DATA_DIR}/tasks" \
    --representation-type cls \
    --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
    --run-name "${RUN_NAME}" \
    --results-file results_final/KNN_val_ITS_aux_weight_ablation_RESULTS.txt
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}