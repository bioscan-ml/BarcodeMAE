#!/bin/bash
# ============================================================================
# ITS-5M Leakage-Free KNN Evaluation — SOFT (distance-weighted) vote variant.
#
# Identical grid-building logic to its_knn_clean_eval.sh (same checkpoints,
# same representations), just adds --knn-weights distance to the
# knn_its_clean.py call. Results auto-route to a separate file
# (KNN_distance_ITS_CLEAN_RESULTS.txt) via knn_results_path() in
# evaluation.py, so this never collides with the existing uniform-vote
# results.
#
# Covers all 35 indices (0-34) -- all checkpoints, including index 32
# (random baseline / transformer / nocls / tokens, which failed once from a
# GPU hardware fault but has since completed on the uniform-vote pass).
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2,3}_tasks.csv) -- same requirement as
# its_knn_clean_eval.sh.
#
# Results: results_final/KNN_distance_ITS_CLEAN_RESULTS.txt
#
# Submit:  sbatch its_knn_clean_softeval.sh
# ============================================================================
#SBATCH --job-name=its_knn_clean_soft
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=0-34%6
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

MAIN_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/${DATASET}"
MINING_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/triplet_mining/${DATASET}"

K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Build the (ckpt|arch|label|repr) grid — identical to its_knn_clean_eval.sh ──
GRID_CKPT=(); GRID_ARCH=(); GRID_LABEL=(); GRID_REPR=()

add_config () {
    local ckpt="$1" arch="$2" label="$3"
    local reprs
    if [ "${label}" = "nocls" ]; then reprs=("tokens"); else reprs=("tokens" "cls" "tokens_with_cls"); fi
    for r in "${reprs[@]}"; do
        GRID_CKPT+=("${ckpt}"); GRID_ARCH+=("${arch}"); GRID_LABEL+=("${label}"); GRID_REPR+=("${r}")
    done
}

add_single () {
    GRID_CKPT+=("$1"); GRID_ARCH+=("$2"); GRID_LABEL+=("$3"); GRID_REPR+=("$4")
}

# Main checkpoints (fungi_its_final.sh)
for ARCH in "maelm" "transformer"; do
    if [ "${ARCH}" = "maelm" ]; then RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_${ARCH}"; CKPT_FILE="checkpoint_encoder.pt"
    else RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}"; CKPT_FILE="checkpoint.pt"; fi
    for LABEL in "nocls" "cls_none" "cls_binary" "cls_triplet" "cls_ce"; do
        RUN_NAME="final_its_${RUN_SUFFIX}_${LABEL}"
        add_config "${MAIN_CKPT_ROOT}/${RUN_NAME}/${CKPT_FILE}" "${ARCH}" "${LABEL}"
    done
done

# Mining ablation (ablation_triplet_mining.sh, random-mining triplet)
for ARCH in "maelm" "transformer"; do
    if [ "${ARCH}" = "maelm" ]; then RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_${ARCH}"; CKPT_FILE="checkpoint_encoder.pt"
    else RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}"; CKPT_FILE="checkpoint.pt"; fi
    RUN_NAME="abl_miningrandom_${RUN_SUFFIX}_cls_triplet"
    add_config "${MINING_CKPT_ROOT}/${RUN_NAME}/${CKPT_FILE}" "${ARCH}" "cls_triplet_miningrandom"
done

# Random-init baseline (transformer only) -- matches its_knn_clean_eval.sh's
# 3-task grid exactly: nocls/tokens, cls/cls, cls/tokens_with_cls.
add_single "RANDOM" "transformer" "nocls" "tokens"
add_single "RANDOM" "transformer" "cls"   "cls"
add_single "RANDOM" "transformer" "cls"   "tokens_with_cls"

TOTAL=${#GRID_CKPT[@]}
echo "Grid has ${TOTAL} entries (expected 35)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= grid size ${TOTAL} — nothing to do."
    exit 0
fi

CKPT="${GRID_CKPT[$SLURM_ARRAY_TASK_ID]}"
ARCH="${GRID_ARCH[$SLURM_ARRAY_TASK_ID]}"
LABEL="${GRID_LABEL[$SLURM_ARRAY_TASK_ID]}"
REPR="${GRID_REPR[$SLURM_ARRAY_TASK_ID]}"

if [ "${CKPT}" = "RANDOM" ]; then
    RUN_NAME="softknnclean_random_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_${LABEL}_${REPR}"
    USE_CLS_ARGS=()
    [ "${LABEL}" = "cls" ] && USE_CLS_ARGS=(--use-cls-token)
    echo "Task: RANDOM baseline | Arch: ${ARCH} | Repr: ${REPR} | Run: ${RUN_NAME}"
    python barcodebert/knn_its_clean.py \
        --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${K_MER} --n-layers ${N_LAYERS} --n-heads ${N_HEADS} \
        --encoder-embed-dim 768 "${USE_CLS_ARGS[@]}" \
        --representation-type "${REPR}" --n-neighbors 1 3 5 7 --metric cosine \
        --knn-weights distance \
        --run-name "${RUN_NAME}" --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt \
        --log-wandb --wandb-project barcodemae_cls
    EC=$?
else
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1
    RUN_NAME="softknnclean_${ARCH}_${LABEL}_${REPR}"
    echo "Task: ${ARCH} | Label: ${LABEL} | Repr: ${REPR} | Ckpt: ${CKPT} | Run: ${RUN_NAME}"
    python barcodebert/knn_its_clean.py \
        --pretrained-checkpoint "${CKPT}" \
        --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --representation-type "${REPR}" --n-neighbors 1 3 5 7 --metric cosine \
        --knn-weights distance \
        --run-name "${RUN_NAME}" --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt \
        --log-wandb --wandb-project barcodemae_cls
    EC=$?
fi

[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
