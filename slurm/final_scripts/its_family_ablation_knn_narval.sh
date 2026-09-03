#!/bin/bash
# ============================================================================
# ITS-5M family-level auxiliary-objective ablation — KNN eval ONLY (k=1,
# leakage-free/genus-level protocol), against checkpoints already trained by
# ablation_taxonomy_level.sh (--export=DATASET=ITS-5M). That script trains
# these checkpoints but deliberately skips eval for ITS-5M (species-level
# knn_its.py is superseded by knn_its_clean.py) -- this script fills that in.
#
# Narval counterpart to its_family_ablation_knn.sh, submitted here because
# fir is under maintenance. Two differences from the fir version:
#   1. gres switched h100 -> a100 (narval's def-lila-ab allocation this
#      session has consistently been a100, not h100).
#   2. --array trimmed to 0-14 (was 0-17): indices 15-17 are the
#      transformer/ce/{tokens,cls,tokens_with_cls} tasks, and that checkpoint
#      (abl_taxafamily_k6_6L6H_transformer_cls_ce) was never actually trained
#      on fir -- its checkpoint dir was empty on transfer, not a copy error.
#      Re-run its_family_ablation_knn.sh --array=15-17 once that checkpoint
#      exists (needs ablation_taxonomy_level.sh --export=DATASET=ITS-5M
#      first) rather than adding it back here.
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv) and the 6 trained checkpoints
# rsync'd over from fir into main_checkpoints_final/ablations/taxonomy_level/
# ITS-5M/ (already done).
#
# Results: results_final/KNN_ITS_family_k1.txt
#
# Submit everything:      sbatch slurm/final_scripts/its_family_ablation_knn_narval.sh
# Submit specific tasks:  sbatch --array=1,2,4,5,7,8,10,11,13,14 slurm/final_scripts/its_family_ablation_knn_narval.sh
# ============================================================================
#SBATCH --job-name=its_family_knn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/taxonomy_level/${DATASET}"
K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Grid (18 tasks): matches ablation_taxonomy_level.sh's family-level
# configs x the 3 representations, one representation per task -- only
# indices 0-14 are submitted here (15-17 need the not-yet-trained
# transformer/ce checkpoint) ────────────────────────────────────────────────
ARCHS=(); AUX_TASKS=(); REPRS_GRID=()
for A in "maelm" "transformer"; do
    for T in "binary" "triplet" "ce"; do
        for R in "tokens" "cls" "tokens_with_cls"; do
            ARCHS+=("${A}"); AUX_TASKS+=("${T}"); REPRS_GRID+=("${R}")
        done
    done
done

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
REPR="${REPRS_GRID[$SLURM_ARRAY_TASK_ID]}"

if [ "${ARCH}" = "maelm" ]; then
    RUN_NAME="abl_taxafamily_k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_${ARCH}_cls_${AUX_TASK}"
    CFILE="checkpoint_encoder.pt"
else
    RUN_NAME="abl_taxafamily_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_cls_${AUX_TASK}"
    CFILE="checkpoint.pt"
fi

CKPT="${CKPT_ROOT}/${RUN_NAME}/${CFILE}"
echo "Arch: ${ARCH} | Aux: ${AUX_TASK} | Repr: ${REPR} | Ckpt: ${CKPT}"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type "${REPR}" --n-neighbors 1 --metric cosine \
    --run-name "knnclean_family_${ARCH}_${AUX_TASK}_${REPR}" \
    --results-file results_final/KNN_ITS_family_k1.txt \
    --log-wandb --wandb-project barcodemae_cls
EC=$?
[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for ${REPR}"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}