#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set genus-level KNN eval for BIOSCAN-5M CE
# aux-weight ablation checkpoints (0.01/0.05/0.50/1.00) plus the w=0.10
# main-sweep checkpoint, on the cluster whose repo root is
# /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE (same cluster as
# its_binary_val_eval.sh). Only run for aux tasks that have actually finished
# training (35/35 epochs) -- binary/triplet were still at epoch 5-6 as of
# the last check, so only ce is included here.
#
# Plain loop, not an sbatch job -- run directly:
#   bash slurm/final_scripts/bioscan5m_ce_val_eval.sh
# ============================================================================

ABL_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/BIOSCAN-5M"
MAIN_CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/BIOSCAN-5M/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
RESULTS_FILE="results_final/KNN_val_bioscan5m_aux_weight_ablation_RESULTS.txt"

for WEIGHT in 0.01 0.05 0.50 1.00; do
    echo "=== ce w=${WEIGHT} | $(date) ==="
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${ABL_BASE}/ablw_bioscan5m_k6_6L6H_6DL6DH_maelm_cls_ce_w${WEIGHT}/checkpoint_encoder.pt" \
        --dataset BIOSCAN-5M --data-dir "${DATA_DIR}" --query-file supervised_val.csv \
        --representation_type cls --taxon genus \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "val_ablw_bioscan5m_ce_w${WEIGHT}" \
        --results-file "${RESULTS_FILE}"
    EC=$?
    [ ${EC} -ne 0 ] && echo "ERROR: w=${WEIGHT} failed (exit ${EC})"
done

echo "=== ce w=0.10 (main) | $(date) ==="
if [ -f "${MAIN_CKPT}" ]; then
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${MAIN_CKPT}" \
        --dataset BIOSCAN-5M --data-dir "${DATA_DIR}" --query-file supervised_val.csv \
        --representation_type cls --taxon genus \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "val_final_bioscan5m_ce_w0.10" \
        --results-file "${RESULTS_FILE}"
    EC=$?
    [ ${EC} -ne 0 ] && echo "ERROR: main w=0.10 failed (exit ${EC})"
else
    echo "ERROR: main checkpoint not found: ${MAIN_CKPT}"
fi

echo "All done at: $(date)"
