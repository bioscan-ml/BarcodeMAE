#!/bin/bash
# ============================================================================
# Ablation: Triplet Loss Margin Sweep
#
# 2 archs × 4 margins = 8 tasks
#   tasks 0-3: maelm       + margins [0.0, 0.1, 0.3, 0.5]
#   tasks 4-7: transformer + margins [0.0, 0.1, 0.3, 0.5]
#
# margin=0.0 → softplus (no free zone, every hard pair contributes)
# margin>0.0 → hinge    (loss = max(0, d_pos - d_neg + margin))
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M ablation_triplet_margin.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     ablation_triplet_margin.sh
# ============================================================================
#SBATCH --job-name=abl_triplet_margin
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%4
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

WANDB_PROJECT="barcodemae_cls"

# ── Grid ──────────────────────────────────────────────────────────────────────
ARCHS=(   "maelm" "maelm" "maelm" "maelm" "transformer" "transformer" "transformer" "transformer")
MARGINS=( 0.0     0.1     0.3     0.5     0.0           0.1           0.3           0.5          )

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
TRIPLET_MARGIN="${MARGINS[$SLURM_ARRAY_TASK_ID]}"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
fi

# ── Fixed config ──────────────────────────────────────────────────────────────
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=35; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; TAXA="genus"

MARGIN_STR=$(echo "${TRIPLET_MARGIN}" | tr '.' 'p')  # 0.3 → 0p3

# ── Naming ────────────────────────────────────────────────────────────────────
if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="abl_margin${MARGIN_STR}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_triplet"
else
    RUN_NAME="abl_margin${MARGIN_STR}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_cls_triplet"
fi

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/triplet_margin/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

echo "Arch: ${ARCH} | Margin: ${TRIPLET_MARGIN} | Dataset: ${DATASET} | Run: ${RUN_NAME}"

# ── Pretraining ───────────────────────────────────────────────────────────────
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
    --use-cls-token --aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN}
    --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    --taxonomy-level ${TAXA} --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
)
[ "${ARCH}" = "maelm" ] && PRETRAIN_ARGS+=(
    --decoder-n-layers "${N_DEC_LAYERS}" --decoder-n-heads "${N_DEC_HEADS}"
    --checkpoint_maelm "${CHECKPOINT_ENC}"
)

echo "=== PRETRAINING ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
[ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
echo "Pretraining done at: $(date)"

[ "${ARCH}" = "maelm" ] && EVAL_CKPT="${CHECKPOINT_ENC}" || EVAL_CKPT="${CHECKPOINT}"
[ ! -f "${EVAL_CKPT}" ] && echo "ERROR: no eval checkpoint" && exit 1

OVERALL_EXIT=0

if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    for REP_TYPE in "tokens" "cls" "tokens_with_cls"; do
        python barcodebert/knn_probing.py \
            --pretrained-checkpoint "${EVAL_CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 1 3 5 7 \
            --run-name "knn_${RUN_NAME}_${REP_TYPE}" \
            --results-file results_final/KNN_RESULTS_final_abl_margin.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}

        python barcodebert/zsc_evaluation_v2.py \
            --pretrained-checkpoint "${EVAL_CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 15 --metric cosine \
            --run-name "zsc_${RUN_NAME}_${REP_TYPE}" \
            --results-file results_final/ZSC_RESULTS_final_abl_margin.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}
    done
else
    FT_TAXA="species"; FT_LR=0.00008; FT_EPOCHS=12; FT_WD=0.00001; FT_BATCH=64
    for REPR in "tokens" "cls" "tokens_with_cls"; do
        FT_RUN="${RUN_NAME}_ft_${FT_TAXA}_${REPR}_ep${FT_EPOCHS}"
        FT_REPR_DIR="${CKPT_BASE}/finetune/${REPR}"
        mkdir -p "${FT_REPR_DIR}"
        torchrun --standalone --nproc_per_node=1 barcodebert/finetuning.py \
            --run-name "${FT_RUN}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --pretrained-checkpoint "${EVAL_CKPT}" --checkpoint "${FT_REPR_DIR}/${FT_RUN}.pt" \
            --taxonomic-level "${FT_TAXA}" --representation-type "${REPR}" \
            --batch-size ${FT_BATCH} --lr ${FT_LR} --weight-decay ${FT_WD} \
            --epochs ${FT_EPOCHS} --max-norm 0.5 --mixed-precision --save-best-model \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}
    done
fi

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}