#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Chains pretraining → KNN evaluation for all 3 CLS aux-loss variants.
#
# Usage (run from BarcodeMAE/):
#   bash slurm/submit_cls_aux.sh
#
# What it does:
#   1. Submits cls_aux_losses.sh  (array 0-2, one job per aux loss type)
#   2. Submits knn_cls_aux.sh     (array 0-2) with --dependency=afterok:<ID>
#      so KNN only starts once ALL pretraining tasks have finished successfully.
# ─────────────────────────────────────────────────────────────────────────────

set -e  # abort on any error

PRETRAIN_SCRIPT="slurm/cls_aux_losses.sh"
KNN_SCRIPT="slurm/knn_cls_aux.sh"

# Submit pretraining and capture the job ID with --parsable
PRETRAIN_JOB_ID=$(sbatch --parsable "${PRETRAIN_SCRIPT}")
echo "Submitted pretraining array job: ${PRETRAIN_JOB_ID}"
echo "  Tasks: triplet (0), supcon (1), ce (2)"

# Submit KNN evaluation to run only after ALL pretraining tasks succeed
KNN_JOB_ID=$(sbatch --parsable --dependency=afterok:${PRETRAIN_JOB_ID} "${KNN_SCRIPT}")
echo "Submitted KNN evaluation array job: ${KNN_JOB_ID}"
echo "  Dependency: afterok:${PRETRAIN_JOB_ID} (all 3 tasks must succeed)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f final_logs/${PRETRAIN_JOB_ID}/${PRETRAIN_JOB_ID}_0.out"