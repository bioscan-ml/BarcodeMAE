#!/bin/bash
# ============================================================================
# ITS-5M Leakage-Free KNN Evaluation — taxonomy-level ablation checkpoints.
# UNIFORM (hard-vote) variant, matching its_knn_clean_eval.sh's default
# (--knn-weights defaults to "uniform" in knn_its_clean.py, so it's simply
# omitted here). Results auto-route to results_final/KNN_ITS_CLEAN_RESULTS.txt
# alongside the main-checkpoint / mining-ablation / random-baseline runs.
#
# Covers the taxonomy_level ablation: does supervising the CLS objective at
# a coarser taxonomic level (family vs order) change KNN transfer quality,
# across CLS objectives (cls_binary/cls_ce/cls_triplet)? 2 levels x 2 arches
# (maelm/transformer) x 3 objectives x 3 representations
# (tokens/cls/tokens_with_cls) = 36 array tasks.
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2,3}_tasks.csv) -- same requirement as
# its_knn_clean_eval.sh.
#
# Results: results_final/KNN_ITS_CLEAN_RESULTS.txt
#
# Submit:  sbatch its_knn_taxonomy_ablation_eval.sh
# ============================================================================
#SBATCH --job-name=its_knn_taxa_abl
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-35%6
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

TAXA_CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/taxonomy_level/${DATASET}"

K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Build the (ckpt|arch|level|label|repr) grid ────────────────────────────
GRID_CKPT=(); GRID_ARCH=(); GRID_LEVEL=(); GRID_LABEL=(); GRID_REPR=()

add_config () {
    # All three objectives here (cls_binary/cls_ce/cls_triplet) are
    # CLS-enabled -> all three representations.
    local ckpt="$1" arch="$2" level="$3" label="$4"
    for r in "tokens" "cls" "tokens_with_cls"; do
        GRID_CKPT+=("${ckpt}"); GRID_ARCH+=("${arch}"); GRID_LEVEL+=("${level}"); GRID_LABEL+=("${label}"); GRID_REPR+=("${r}")
    done
}

for LEVEL in "taxafamily" "taxaorder"; do
    for ARCH in "maelm" "transformer"; do
        if [ "${ARCH}" = "maelm" ]; then RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_${ARCH}"; CKPT_FILE="checkpoint_encoder.pt"
        else RUN_SUFFIX="k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}"; CKPT_FILE="checkpoint.pt"; fi
        for LABEL in "cls_binary" "cls_ce" "cls_triplet"; do
            RUN_NAME="abl_${LEVEL}_${RUN_SUFFIX}_${LABEL}"
            add_config "${TAXA_CKPT_ROOT}/${RUN_NAME}/${CKPT_FILE}" "${ARCH}" "${LEVEL}" "${LABEL}"
        done
    done
done

TOTAL=${#GRID_CKPT[@]}
echo "Grid has ${TOTAL} entries (expected 36)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= grid size ${TOTAL} — nothing to do."
    exit 0
fi

CKPT="${GRID_CKPT[$SLURM_ARRAY_TASK_ID]}"
ARCH="${GRID_ARCH[$SLURM_ARRAY_TASK_ID]}"
LEVEL="${GRID_LEVEL[$SLURM_ARRAY_TASK_ID]}"
LABEL="${GRID_LABEL[$SLURM_ARRAY_TASK_ID]}"
REPR="${GRID_REPR[$SLURM_ARRAY_TASK_ID]}"

[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1
RUN_NAME="knnclean_${LEVEL}_${ARCH}_${LABEL}_${REPR}"
echo "Task: ${ARCH} | Level: ${LEVEL} | Label: ${LABEL} | Repr: ${REPR} | Ckpt: ${CKPT} | Run: ${RUN_NAME}"
python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type "${REPR}" --n-neighbors 1 3 5 7 --metric cosine \
    --run-name "${RUN_NAME}" --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt \
    --log-wandb --wandb-project barcodemae_cls
EC=$?

[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}