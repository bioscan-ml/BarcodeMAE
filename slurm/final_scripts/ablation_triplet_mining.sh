#!/bin/bash
# ============================================================================
# Ablation: Triplet Mining Strategy (uniform-random vs batch-hard)
#
# 2 archs × 1 mining strategy (random) = 2 tasks
#   tasks 0-1: maelm, transformer  + mining=random
#
# batch-hard is NOT re-run here — it's already covered by the main triplet
# experiments (bioscan5m_final.sh task 3/8, fungi_its_final.sh task 3/8).
# Compare against those runs' KNN_RESULTS_final.txt / test_results_species.json.
#
# Everything else (margin, taxonomy level, weight, schedule) matches the main
# triplet experiments exactly, so results are directly comparable.
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M ablation_triplet_mining.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     ablation_triplet_mining.sh
# ============================================================================
#SBATCH --job-name=abl_triplet_mining
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
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
ARCHS=("maelm" "transformer")
MINING="random"

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
    EPOCHS=15
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
    EPOCHS=35
fi

# ── Fixed config (matches main triplet experiments exactly) ──────────────────
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; TAXA="genus"
TRIPLET_MARGIN=0.0

# ── Naming ────────────────────────────────────────────────────────────────────
if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="abl_mining${MINING}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_triplet"
else
    RUN_NAME="abl_mining${MINING}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_cls_triplet"
fi

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/triplet_mining/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

echo "Arch: ${ARCH} | Mining: ${MINING} | Dataset: ${DATASET} | Run: ${RUN_NAME}"

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
    --use-cls-token --aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN} --triplet-mining ${MINING}
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
            --results-file results_final/KNN_RESULTS_final_abl_mining.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}

        python barcodebert/zsc_evaluation_v2.py \
            --pretrained-checkpoint "${EVAL_CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 15 --metric cosine \
            --run-name "zsc_${RUN_NAME}_${REP_TYPE}" \
            --results-file results_final/ZSC_RESULTS_final_abl_mining.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}
    done
else
    for REPR in "tokens" "cls" "tokens_with_cls"; do
        python barcodebert/knn_its.py \
            --pretrained-checkpoint "${EVAL_CKPT}" \
            --data-dir              "${DATA_DIR}" \
            --run-name              "knn_its_${RUN_NAME}_${REPR}" \
            --n-neighbors           1 3 5 7 \
            --metric                cosine \
            --representation-type   "${REPR}" \
            --results-file          results_final/KNN_ITS_RESULTS_final_abl_mining.txt \
            --log-wandb \
            --wandb-project         "${WANDB_PROJECT}"
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}
    done
fi

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}