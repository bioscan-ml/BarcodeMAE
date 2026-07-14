#!/bin/bash
# ============================================================================
# Random-Initialized Encoder Baseline — KNN, no pretraining
#
# Sanity-check baseline: run KNN eval with an UNTRAINED encoder to confirm
# that pretrained checkpoints actually learn something beyond random
# projections. Uses random_knn.py (BIOSCAN-5M) / random_knn_its.py (fungi
# ITS-5M). Architecture: transformer only (untrained BertForTokenClassification,
# head stripped).
#
# 3 tasks (0-2):
#   0: nocls,             representation_type=tokens
#   1: --use-cls-token,   representation_type=cls
#   2: --use-cls-token,   representation_type=tokens_with_cls
# Each task builds its own fresh random model — cheap (no training), so
# there's no cost concern running these as separate tasks/models per
# representation type.
#
# DATASET is an env var like the rest of slurm/final_scripts/*.sh.
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M random_baseline_knn.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     random_baseline_knn.sh
#
# Results: results_final/RANDOM_KNN_RESULTS.txt (BIOSCAN-5M)
#          results_final/RANDOM_KNN_ITS_RESULTS.txt (ITS-5M)
#
# Time budget: representations_from_df / extract_representations embed ONE
# sequence at a time (no batching), so this is slower than the row count
# alone suggests. 2h was too tight (hit the wall on the ITS-5M run). Bumped
# to 8h. Unlike pretraining there's no resume/checkpoint here — a timeout
# loses the whole embedding pass, so this needs headroom.
# ============================================================================
#SBATCH --job-name=random_baseline_knn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Dataset: ${DATASET:-BIOSCAN-5M} | $(date)"

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

# ── Grid ──────────────────────────────────────────────────────────────────────
ARCH="transformer"
CLS_LABELS=("nocls"  "cls" "cls"            )
REPR_TYPES=("tokens" "cls" "tokens_with_cls")

CLS_LABEL="${CLS_LABELS[$SLURM_ARRAY_TASK_ID]}"
REPR_TYPE="${REPR_TYPES[$SLURM_ARRAY_TASK_ID]}"
USE_CLS_ARGS=()
[ "${CLS_LABEL}" = "cls" ] && USE_CLS_ARGS=(--use-cls-token)

# ── Dataset (overridable via --export=DATASET=...) ────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
fi

K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; ENCODER_DIM=768

RUN_NAME="random_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_${CLS_LABEL}_${REPR_TYPE}"

echo "Dataset: ${DATASET} | Arch: ${ARCH} | CLS: ${CLS_LABEL} | Repr: ${REPR_TYPE} | Run: ${RUN_NAME}"

if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    python barcodebert/random_knn.py \
        --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE} \
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS} --encoder-embed-dim ${ENCODER_DIM} \
        "${USE_CLS_ARGS[@]}" \
        --taxon genus --n-neighbors 1 3 5 7 --representation-type "${REPR_TYPE}" \
        --run-name "${RUN_NAME}" \
        --results-file results_final/RANDOM_KNN_RESULTS.txt
    EC=$?
else
    python barcodebert/random_knn_its.py \
        --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE} \
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS} --encoder-embed-dim ${ENCODER_DIM} \
        "${USE_CLS_ARGS[@]}" \
        --n-neighbors 1 3 5 7 --representation-type "${REPR_TYPE}" \
        --run-name "${RUN_NAME}" \
        --results-file results_final/RANDOM_KNN_ITS_RESULTS.txt
    EC=$?
fi

[ ${EC} -ne 0 ] && echo "ERROR: random baseline KNN failed"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}