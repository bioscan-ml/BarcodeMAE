#!/bin/bash
# ============================================================================
# Leakage-free VALIDATION-set KNN eval for the aux-loss-weight sweep, for the
# chosen task per dataset (BIOSCAN-5M: CE, ITS-5M: binary) -- used to pick
# the weight WITHOUT touching unseen.csv/test1/test2, which are the same
# files the paper's headline results are computed on.
#
# 10 array tasks (0-9): 5 weights (0.01, 0.05, 0.10-main, 0.50, 1.00) x 2
# datasets. Uses knn_probing.py --query-file supervised_val.csv for
# BIOSCAN-5M, and the new knn_its_clean_val.py (genus-level scoring on
# trainset_valid.fasta's clean species_level specimens) for ITS-5M.
#
# REQUIRES data/ITS-5M/tasks/trainset_valid_tasks.csv already exported:
#   python barcodebert/analyze_its_valtrain_overlap.py --data-dir data/ITS-5M --export-dir data/ITS-5M/tasks
#
# *** FIX THESE TWO PATHS BEFORE SUBMITTING *** -- the main-sweep (w=0.10)
# checkpoints live outside the ablation tree; the ITS-5M one in particular
# was NOT at either path guessed earlier in this repo's history, so this is
# a placeholder guess. Find yours with, e.g.:
#   find /home/m4safari/projects/def-lila-ab/m4safari -iname "final_k6_6L6H*cls_ce*" -type d
#   find /home/m4safari/projects/def-lila-ab/m4safari -iname "final_its_k6_6L6H*cls_binary*" -type d
MAIN_CKPT_BIOSCAN_CE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/BIOSCAN-5M/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt"
MAIN_CKPT_ITS_BINARY="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ITS-5M/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/checkpoint_encoder.pt"
# ============================================================================
#SBATCH --job-name=aux_weight_val_eval
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
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

# ── Grid (10 tasks): 5 weights x 2 datasets ──────────────────────────────────
DATASETS=("bioscan" "bioscan" "bioscan" "bioscan" "bioscan"   "its" "its" "its" "its" "its")
WEIGHTS=(  0.01      0.05      0.10      0.50      1.00        0.01  0.05  0.10  0.50  1.00)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

echo "dataset: ${DATASET} | weight: ${WEIGHT}"

if [ "${DATASET}" = "bioscan" ]; then
    if [ "${WEIGHT}" = "0.10" ]; then
        CKPT="${MAIN_CKPT_BIOSCAN_CE}"
        RUN_NAME="val_final_bioscan5m_ce_w0.10"
    else
        CKPT="${ABL_BASE_BIOSCAN}/ablw_bioscan5m_k6_6L6H_6DL6DH_maelm_cls_ce_w${WEIGHT}/checkpoint_encoder.pt"
        RUN_NAME="val_ablw_bioscan5m_ce_w${WEIGHT}"
    fi
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CKPT}" \
        --dataset BIOSCAN-5M --data-dir "${DATA_DIR_BIOSCAN}" --query-file supervised_val.csv \
        --representation_type cls --taxon genus \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "${RUN_NAME}" \
        --results-file results_final/KNN_val_bioscan5m_aux_weight_ablation_RESULTS.txt
    EC=$?
else
    if [ "${WEIGHT}" = "0.10" ]; then
        CKPT="${MAIN_CKPT_ITS_BINARY}"
        RUN_NAME="val_final_its_binary_w0.10"
    else
        CKPT="${ABL_BASE_ITS}/ablw_its_k6_6L6H_6DL6DH_maelm_cls_binary_w${WEIGHT}/checkpoint_encoder.pt"
        RUN_NAME="val_ablw_its_binary_w${WEIGHT}"
    fi
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

    python barcodebert/knn_its_clean_val.py \
        --pretrained-checkpoint "${CKPT}" \
        --data-dir "${DATA_DIR_ITS}" --tasks-dir "${DATA_DIR_ITS}/tasks" \
        --representation-type cls \
        --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine --knn-weights uniform \
        --run-name "${RUN_NAME}" \
        --results-file results_final/KNN_val_ITS_aux_weight_ablation_RESULTS.txt
    EC=$?
fi

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
