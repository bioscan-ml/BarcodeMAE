#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set KNN eval for the aux-loss-weight sweep, for the
# NON-chosen tasks per dataset: BIOSCAN-5M binary+triplet, ITS-5M ce+triplet
# (the chosen tasks -- BIOSCAN-5M/CE, ITS-5M/binary -- are covered by
# aux_weight_val_eval.sh instead). No w=0.10 main-sweep row here, only the
# 4 ablation weights per task, matching what was actually requested.
#
# 16 array tasks (0-15): 4 weights (0.01/0.05/0.5/1.0) x 4 (task, dataset)
# combos: BIOSCAN-5M binary, BIOSCAN-5M triplet, ITS-5M ce, ITS-5M triplet.
#
# REQUIRES data/ITS-5M/tasks/trainset_valid_tasks.csv already exported:
#   python barcodebert/analyze_its_valtrain_overlap.py --data-dir data/ITS-5M --export-dir data/ITS-5M/tasks
# ============================================================================
#SBATCH --job-name=aux_weight_val_eval_othertasks
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-15
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

DATA_DIR_BIOSCAN="data/BIOSCAN-5M"
DATA_DIR_ITS="data/ITS-5M"
ABL_BASE_BIOSCAN="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/BIOSCAN-5M"
ABL_BASE_ITS="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/ITS-5M"

# ── Grid (16 tasks): 4 weights x 4 (dataset, task) combos ───────────────────
DATASETS=("bioscan" "bioscan" "bioscan" "bioscan"   "bioscan" "bioscan" "bioscan" "bioscan"   "its" "its" "its" "its"   "its" "its" "its" "its")
TASKS=(    "binary"  "binary"  "binary"  "binary"    "triplet" "triplet" "triplet" "triplet"   "ce"  "ce"  "ce"  "ce"    "triplet" "triplet" "triplet" "triplet")
WEIGHTS=(   0.01      0.05      0.5       1.0         0.01      0.05      0.5       1.0         0.01  0.05  0.5   1.0     0.01      0.05      0.5       1.0)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

echo "dataset: ${DATASET} | aux_task: ${AUX_TASK} | weight: ${WEIGHT}"

if [ "${DATASET}" = "bioscan" ]; then
    CKPT="${ABL_BASE_BIOSCAN}/ablw_bioscan5m_k6_6L6H_6DL6DH_maelm_cls_${AUX_TASK}_w${WEIGHT}/checkpoint_encoder.pt"
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CKPT}" \
        --dataset BIOSCAN-5M --data-dir "${DATA_DIR_BIOSCAN}" --query-file supervised_val.csv \
        --representation_type cls --taxon genus \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "val_ablw_bioscan5m_${AUX_TASK}_w${WEIGHT}" \
        --results-file results_final/KNN_val_bioscan5m_aux_weight_ablation_RESULTS.txt
    EC=$?
else
    CKPT="${ABL_BASE_ITS}/ablw_its_k6_6L6H_6DL6DH_maelm_cls_${AUX_TASK}_w${WEIGHT}/checkpoint_encoder.pt"
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

    python barcodebert/knn_its_clean_val.py \
        --pretrained-checkpoint "${CKPT}" \
        --data-dir "${DATA_DIR_ITS}" --tasks-dir "${DATA_DIR_ITS}/tasks" \
        --representation-type cls \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "val_ablw_its_${AUX_TASK}_w${WEIGHT}" \
        --results-file results_final/KNN_val_ITS_aux_weight_ablation_RESULTS.txt
    EC=$?
fi

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
