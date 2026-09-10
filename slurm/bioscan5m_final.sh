#!/bin/bash
# ============================================================================
# BIOSCAN-5M Final Experiments — Pretraining + KNN + ZSC
#
# 10 array tasks (0-9), each covers one pretrain + inline eval:
#
#   Task | Arch        | CLS | Aux task
#   -----|-------------|-----|--------------------
#     0  | maelm       | no  | none
#     1  | maelm       | yes | none  (CLS baseline)
#     2  | maelm       | yes | binary (BCE)
#     3  | maelm       | yes | triplet
#     4  | maelm       | yes | genus CE
#     5  | transformer | no  | none
#     6  | transformer | yes | none  (CLS baseline)
#     7  | transformer | yes | binary (BCE)
#     8  | transformer | yes | triplet
#     9  | transformer | yes | genus CE
#
# KNN / ZSC representation types per model:
#   no-CLS models  → tokens
#   CLS models     → tokens | cls | tokens_with_cls
#
# Checkpoints saved under: main_checkpoints_final/BIOSCAN-5M/
# ============================================================================
#SBATCH --job-name=final_bioscan5m
#SBATCH --account=<your_slurm_account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%4
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Node $SLURMD_NODENAME | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

echo "Python: $(which python) — $(python --version)"

export WANDB_MODE=offline
export WANDB_DIR="${REPO_DIR:-$HOME/BarcodeMAE-plus}/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

# ── Sweep grid ────────────────────────────────────────────────────────────────
ARCHS=(    "maelm"  "maelm"  "maelm"  "maelm"  "maelm"  "transformer" "transformer" "transformer" "transformer" "transformer")
HAS_CLS=(  "no"     "yes"    "yes"    "yes"    "yes"    "no"          "yes"         "yes"         "yes"         "yes"        )
AUX_TASKS=("none"   "none"   "binary" "triplet" "ce"    "none"        "none"        "binary"      "triplet"     "ce"         )

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
HAS_CLS_VAL="${HAS_CLS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed pretraining config ──────────────────────────────────────────────────
DATASET="BIOSCAN-5M"
DATA_DIR="${REPO_DIR:-$HOME/BarcodeMAE-plus}/data/${DATASET}"
K_MER=6;   STRIDE=6
N_LAYERS=6; N_HEADS=6
N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=35
AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
# k=16, m=4 for all aux tasks → equal labeled samples per batch (64/128)
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"
TRIPLET_MARGIN=0.0; CLS_TAXA_LOSS_W=0.1   # triplet: softplus, best from sweep task 4

# ── Run name & checkpoint paths ───────────────────────────────────────────────
if [ "$HAS_CLS_VAL" = "no" ]; then
    CLS_LABEL="nocls"
else
    CLS_LABEL="cls_${AUX_TASK}"
fi

if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="final_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_${CLS_LABEL}"
else
    RUN_NAME="final_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_${CLS_LABEL}"
fi

CKPT_BASE="${REPO_DIR:-$HOME/BarcodeMAE-plus}/main_checkpoints_final/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

echo "=========================================="
echo "Arch: ${ARCH} | CLS: ${HAS_CLS_VAL} | Aux: ${AUX_TASK}"
echo "Run:  ${RUN_NAME}"
echo "=========================================="

# ── Build pretraining args ────────────────────────────────────────────────────
PRETRAIN_ARGS=(
    --run-name           "${RUN_NAME}"
    --dataset            "${DATASET}"
    --data-dir           "${DATA_DIR}"
    --arch               "${ARCH}"
    --k-mer              ${K_MER}
    --stride             ${STRIDE}
    --n-layers           ${N_LAYERS}
    --n-heads            ${N_HEADS}
    --batch-size         ${BATCH_SIZE}
    --lr                 ${LR}
    --weight-decay       ${WD}
    --epochs             ${EPOCHS}
    --mask-token-ratio   ${MASK_TOKEN_RATIO}
    --random-token-ratio ${RANDOM_TOKEN_RATIO}
    --masked-loss-weight ${MASKED_LOSS_WEIGHT}
    --max-norm           0.5
    --separate_loss      true
    --mixed-precision
    --save-best-model
    --log-wandb
    --checkpoint         "${CHECKPOINT}"
)

if [ "${ARCH}" = "maelm" ]; then
    PRETRAIN_ARGS+=(
        --decoder-n-layers "${N_DEC_LAYERS}"
        --decoder-n-heads  "${N_DEC_HEADS}"
        --checkpoint_maelm "${CHECKPOINT_ENC}"
    )
fi

if [ "$HAS_CLS_VAL" = "yes" ]; then
    PRETRAIN_ARGS+=(
        --taxonomy-level         ${TAXA}
        --taxonomy-max-pairs     ${NUM_PAIRS}
        --k-classes              ${K_CLASSES}
        --m-per-class            ${M_PER_CLASS}
        --aux-loss-weight        ${AUX_LOSS_WEIGHT}
        --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    )
    case "${AUX_TASK}" in
        binary)
            PRETRAIN_ARGS+=(
                --use-cls-token
                --enable-cls-taxonomy
                --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}
            )
            ;;
        triplet)
            PRETRAIN_ARGS+=(
                --use-cls-token
                --aux-loss-type  triplet
                --triplet-margin ${TRIPLET_MARGIN}
            )
            ;;
        ce)
            PRETRAIN_ARGS+=(
                --use-cls-token
                --aux-loss-type ce
            )
            ;;
    esac
fi

# ── Pretraining ───────────────────────────────────────────────────────────────
echo "=== PRETRAINING: ${RUN_NAME} ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
PRETRAIN_EXIT=$?
if [ ${PRETRAIN_EXIT} -ne 0 ]; then
    echo "ERROR: Pretraining failed (exit ${PRETRAIN_EXIT})"
    exit ${PRETRAIN_EXIT}
fi
echo "Pretraining finished at: $(date)"

# ── Select evaluation checkpoint ─────────────────────────────────────────────
# MAELM: encoder-only checkpoint; transformer: full checkpoint
if [ "${ARCH}" = "maelm" ]; then
    EVAL_CKPT="${CHECKPOINT_ENC}"
else
    EVAL_CKPT="${CHECKPOINT}"
fi

# Fallback to best-model checkpoint if encoder file is missing
if [ ! -f "${EVAL_CKPT}" ]; then
    echo "WARNING: expected checkpoint not found at ${EVAL_CKPT}, trying best_model.pt"
    EVAL_CKPT="${CKPT_BASE}/best_model.pt"
fi

if [ ! -f "${EVAL_CKPT}" ]; then
    echo "ERROR: No evaluation checkpoint found. Skipping eval."
    exit 1
fi

# ── Representation types for eval ─────────────────────────────────────────────
if [ "$HAS_CLS_VAL" = "no" ]; then
    REP_TYPES=("tokens")
else
    # tokens  : mean of sequence tokens only (no CLS), compares fairly to no-CLS models
    # cls     : CLS token only
    # tokens_with_cls: mean of CLS + all sequence tokens
    REP_TYPES=("tokens" "cls" "tokens_with_cls")
fi

OVERALL_EXIT=0

# ── KNN evaluation ────────────────────────────────────────────────────────────
echo "=== KNN EVALUATION ==="
for REP_TYPE in "${REP_TYPES[@]}"; do
    echo "--- KNN repr=${REP_TYPE} ---"
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${EVAL_CKPT}" \
        --dataset               "${DATASET}" \
        --data-dir              "${DATA_DIR}" \
        --representation_type   "${REP_TYPE}" \
        --taxon                 genus \
        --n-neighbors           1 \
        --run-name              "knn_${RUN_NAME}_${REP_TYPE}" \
        --log-wandb
    EC=$?
    [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for repr=${REP_TYPE} (exit ${EC})" && OVERALL_EXIT=${EC}
done

# ── ZSC evaluation ────────────────────────────────────────────────────────────
echo "=== ZSC EVALUATION ==="
for REP_TYPE in "${REP_TYPES[@]}"; do
    echo "--- ZSC repr=${REP_TYPE} ---"
    python barcodebert/zsc_evaluation_v2.py \
        --pretrained-checkpoint "${EVAL_CKPT}" \
        --dataset               "${DATASET}" \
        --data-dir              "${DATA_DIR}" \
        --representation_type   "${REP_TYPE}" \
        --taxon                 genus \
        --n-neighbors           15 \
        --metric                cosine \
        --run-name              "zsc_${RUN_NAME}_${REP_TYPE}" \
        --log-wandb
    EC=$?
    [ ${EC} -ne 0 ] && echo "ERROR: ZSC failed for repr=${REP_TYPE} (exit ${EC})" && OVERALL_EXIT=${EC}
done

echo "=========================================="
echo "All done at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="
exit ${OVERALL_EXIT}
