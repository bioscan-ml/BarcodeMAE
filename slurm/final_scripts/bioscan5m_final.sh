#!/bin/bash
# ============================================================================
# BIOSCAN-5M Final Experiments — Pretraining + KNN (k=1,3,5,7) + ZSC
#
# 10 array tasks (0-9):
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
# KNN / ZSC repr types:  no-CLS → tokens | CLS → tokens, cls, tokens_with_cls
# KNN neighbors:         k = 1, 3, 5, 7
# Results:               results_final/KNN_RESULTS_final.txt / ZSC_RESULTS_final.txt
# Wandb project:         barcodemae_cls  (offline → wandb_final/)
# Checkpoints:           main_checkpoints_final/BIOSCAN-5M/
# ============================================================================
#SBATCH --job-name=final_bioscan5m
#SBATCH --account=def-lila-ab
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

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

# ── Sweep grid ────────────────────────────────────────────────────────────────
ARCHS=(    "maelm"  "maelm"  "maelm"  "maelm"  "maelm"  "transformer" "transformer" "transformer" "transformer" "transformer")
HAS_CLS=(  "no"     "yes"    "yes"    "yes"    "yes"    "no"          "yes"         "yes"         "yes"         "yes"        )
AUX_TASKS=("none"   "none"   "binary" "triplet" "ce"    "none"        "none"        "binary"      "triplet"     "ce"         )

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
HAS_CLS_VAL="${HAS_CLS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed config ──────────────────────────────────────────────────────────────
DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=35; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"
TRIPLET_MARGIN=0.0; CLS_TAXA_LOSS_W=0.1

# ── Naming ────────────────────────────────────────────────────────────────────
[ "$HAS_CLS_VAL" = "no" ] && CLS_LABEL="nocls" || CLS_LABEL="cls_${AUX_TASK}"
if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="final_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_${CLS_LABEL}"
else
    RUN_NAME="final_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_${CLS_LABEL}"
fi

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"

echo "Arch: ${ARCH} | CLS: ${HAS_CLS_VAL} | Aux: ${AUX_TASK} | Run: ${RUN_NAME}"

# ── Pretraining args ──────────────────────────────────────────────────────────
PRETRAIN_ARGS=(
    --run-name "${RUN_NAME}" --dataset "${DATASET}" --data-dir "${DATA_DIR}"
    --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE}
    --n-layers ${N_LAYERS} --n-heads ${N_HEADS}
    --batch-size ${BATCH_SIZE} --lr ${LR} --weight-decay ${WD}
    --epochs ${EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO}
    --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT}
    --max-norm 0.5 --separate_loss true --mixed-precision
    --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}"
    --checkpoint "${CHECKPOINT}"
)
[ "${ARCH}" = "maelm" ] && PRETRAIN_ARGS+=(
    --decoder-n-layers "${N_DEC_LAYERS}" --decoder-n-heads "${N_DEC_HEADS}"
    --checkpoint_maelm "${CHECKPOINT_ENC}"
)
if [ "$HAS_CLS_VAL" = "yes" ]; then
    PRETRAIN_ARGS+=(
        --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS}
        --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
        --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    )
    case "${AUX_TASK}" in
        none)    PRETRAIN_ARGS+=(--use-cls-token) ;;
        binary)  PRETRAIN_ARGS+=(--use-cls-token --enable-cls-taxonomy --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}) ;;
        triplet) PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN}) ;;
        ce)      PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type ce) ;;
    esac
fi

# ── Pretraining ───────────────────────────────────────────────────────────────
echo "=== PRETRAINING ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
[ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
echo "Pretraining done at: $(date)"

# ── Eval checkpoint ───────────────────────────────────────────────────────────
[ "${ARCH}" = "maelm" ] && EVAL_CKPT="${CHECKPOINT_ENC}" || EVAL_CKPT="${CHECKPOINT}"
[ ! -f "${EVAL_CKPT}" ] && EVAL_CKPT="${CKPT_BASE}/best_model.pt"
[ ! -f "${EVAL_CKPT}" ] && echo "ERROR: no eval checkpoint found" && exit 1

[ "$HAS_CLS_VAL" = "no" ] && REP_TYPES=("tokens") || REP_TYPES=("tokens" "cls" "tokens_with_cls")

OVERALL_EXIT=0

# ── KNN (k = 1, 3, 5, 7) ─────────────────────────────────────────────────────
echo "=== KNN EVALUATION ==="
for REP_TYPE in "${REP_TYPES[@]}"; do
    echo "--- KNN repr=${REP_TYPE} ---"
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${EVAL_CKPT}" \
        --dataset               "${DATASET}" \
        --data-dir              "${DATA_DIR}" \
        --representation_type   "${REP_TYPE}" \
        --taxon                 genus \
        --n-neighbors           1 3 5 7 \
        --run-name              "knn_${RUN_NAME}_${REP_TYPE}" \
        --results-file          results_final/KNN_RESULTS_final.txt \
        --wandb-project         "${WANDB_PROJECT}" \
        --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for ${REP_TYPE}" && OVERALL_EXIT=${EC}
done

# ── ZSC ───────────────────────────────────────────────────────────────────────
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
        --results-file          results_final/ZSC_RESULTS_final.txt \
        --wandb-project         "${WANDB_PROJECT}" \
        --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: ZSC failed for ${REP_TYPE}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}