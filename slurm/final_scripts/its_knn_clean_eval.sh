#!/bin/bash
# ============================================================================
# ITS-5M Leakage-Free KNN Evaluation — main checkpoints + mining ablation +
# random baseline. Each task evaluates BOTH species_level and genus_level in
# one pass (knn_its_clean.py shares embeddings across both — see that file).
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2,3}_tasks.csv).
#
# NOT included yet: the taxonomy-level ablation (family/order/bin) — still
# training as of this writing. Add its checkpoints to the CONFIGS loop below
# once those runs finish; the grid-building logic doesn't need to change.
#
# Grid (built as CKPT|ARCH|LABEL|REPR entries, one array task each):
#   - 10 main checkpoints (fungi_its_final.sh): nocls=1 repr, cls_*=3 reprs
#     each -> 2x1 + 8x3 = 26
#   - 2 mining-ablation checkpoints (ablation_triplet_mining.sh, cls_triplet
#     random mining): 3 reprs each -> 6
#   - 1 random-init baseline (transformer only, per earlier decision): 3
#     configs (nocls/tokens, cls/cls, cls/tokens_with_cls) -> 3
#   Total: 35 array tasks (indices 0-34)
#
# Results: results_final/KNN_ITS_CLEAN_RESULTS.txt
#          one line per (run, task, test_set, k): species_level and
#          genus_level accuracies are both in this same file, distinguished
#          by the "_species_level_" / "_genus_level_" substring in the tag.
#
# Submit:  sbatch its_knn_clean_eval.sh
# ============================================================================
#SBATCH --job-name=its_knn_clean
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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

MAIN_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/${DATASET}"
MINING_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/triplet_mining/${DATASET}"

K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Build the (ckpt|arch|label|repr) grid ──────────────────────────────────────
# CKPT="RANDOM" is the sentinel for the random-init baseline (no checkpoint).
GRID_CKPT=(); GRID_ARCH=(); GRID_LABEL=(); GRID_REPR=()

add_config () {
    # Expands into one grid entry per representation type available for this
    # config (nocls -> tokens only; CLS-enabled -> tokens/cls/tokens_with_cls).
    local ckpt="$1" arch="$2" label="$3"
    local reprs
    if [ "${label}" = "nocls" ]; then reprs=("tokens"); else reprs=("tokens" "cls" "tokens_with_cls"); fi
    for r in "${reprs[@]}"; do
        GRID_CKPT+=("${ckpt}"); GRID_ARCH+=("${arch}"); GRID_LABEL+=("${label}"); GRID_REPR+=("${r}")
    done
}

add_single () {
    # Pushes exactly one grid entry, no repr expansion — used for the random
    # baseline, which needs specific (label, repr) pairs, not a full cross product.
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

# Random-init baseline (transformer only) — matches random_baseline_knn.sh's
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
    RUN_NAME="knnclean_random_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_${LABEL}_${REPR}"
    USE_CLS_ARGS=()
    [ "${LABEL}" = "cls" ] && USE_CLS_ARGS=(--use-cls-token)
    echo "Task: RANDOM baseline | Arch: ${ARCH} | Repr: ${REPR} | Run: ${RUN_NAME}"
    python barcodebert/knn_its_clean.py \
        --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${K_MER} --n-layers ${N_LAYERS} --n-heads ${N_HEADS} \
        --encoder-embed-dim 768 "${USE_CLS_ARGS[@]}" \
        --representation-type "${REPR}" --n-neighbors 1 3 5 7 --metric cosine \
        --run-name "${RUN_NAME}" --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt \
        --log-wandb --wandb-project barcodemae_cls
    EC=$?
else
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1
    RUN_NAME="knnclean_${ARCH}_${LABEL}_${REPR}"
    echo "Task: ${ARCH} | Label: ${LABEL} | Repr: ${REPR} | Ckpt: ${CKPT} | Run: ${RUN_NAME}"
    python barcodebert/knn_its_clean.py \
        --pretrained-checkpoint "${CKPT}" \
        --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --representation-type "${REPR}" --n-neighbors 1 3 5 7 --metric cosine \
        --run-name "${RUN_NAME}" --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt \
        --log-wandb --wandb-project barcodemae_cls
    EC=$?
fi

[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
