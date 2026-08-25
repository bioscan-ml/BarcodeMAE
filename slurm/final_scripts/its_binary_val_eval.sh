#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set genus-level KNN eval for ITS-5M binary
# aux-weight ablation checkpoints (0.01/0.05/0.50/1.00), on the cluster whose
# repo root is /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE
# (not the narval BarcodeMAE_final/BarcodeMAE checkout).
#
# Plain loop, not an sbatch job -- run directly:
#   bash slurm/final_scripts/its_binary_val_eval.sh
#
# REQUIRES data/ITS-5M/tasks/trainset_valid_tasks.csv already exported (via
# analyze_its_valtrain_overlap.py) -- already done for this checkout.
# ============================================================================

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/ITS-5M"

for WEIGHT in 0.01 0.05 0.50 1.00; do
    echo "=== binary w=${WEIGHT} | $(date) ==="
    python barcodebert/knn_its_clean_val.py \
        --pretrained-checkpoint "${CKPT_BASE}/ablw_its_k6_6L6H_6DL6DH_maelm_cls_binary_w${WEIGHT}/checkpoint_encoder.pt" \
        --data-dir "${DATA_DIR}" --tasks-dir "${DATA_DIR}/tasks" \
        --representation-type cls \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "val_ablw_its_binary_w${WEIGHT}" \
        --results-file results_final/KNN_val_ITS_aux_weight_ablation_RESULTS.txt
    EC=$?
    [ ${EC} -ne 0 ] && echo "ERROR: w=${WEIGHT} failed (exit ${EC})"
done

echo "All done at: $(date)"
