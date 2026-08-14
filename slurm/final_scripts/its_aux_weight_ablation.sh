#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) auxiliary-loss-WEIGHT ablation, for the best ITS-5M
# configuration only (encoder-decoder, +CLS+Binary). Sweeps
# --cls-taxonomy-loss-weight (the Binary/BCE auxiliary objective's weight)
# over a short list of values around the main run's 0.1 baseline. 0.1 itself
# is NOT included -- that checkpoint/result already exists from
# fungi_its_final.sh, no need to retrain it.
#
# Unlike fungi_its_final.sh, this is a SINGLE fixed config (maelm, CLS,
# binary) with only the weight varying -- not the full 10-config grid -- and
# it does NOT run finetuning: the paper's KNN numbers use the raw pretrained
# encoder (checkpoint_encoder.pt) directly, so finetuning is unnecessary
# extra cost here. Pretraining is immediately followed by the same
# leakage-free genus-level KNN eval (uniform + softmax, k=1..50) used for the
# main results, so each task is fully self-contained.
#
# REQUIRES its_export_tasks.sh already run (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv).
#
# 4 array tasks (0-3), one per weight value. Each is a FULL pretraining run
# (up to the 28h time limit, matching fungi_its_final.sh) -- this ablation is
# expensive; that's why it's scoped to 4 extra values, not a full sweep.
#
# Checkpoints: main_checkpoints_final/ablations/aux_weight/ITS-5M/
# Results:     results_final/KNN_ITS_aux_weight_ablation_RESULTS.txt
#              (uniform, auto-routed to the _softmax_ variant for softmax)
#
# Submit: sbatch slurm/final_scripts/its_aux_weight_ablation.sh
# ============================================================================
#SBATCH --job-name=its_aux_weight_ablation
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=28:00:00
#SBATCH --array=0-3
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
export WANDB_DIR="/project/6045013/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

# ── Grid (4 tasks): CLS-taxonomy (Binary) loss weight, 0.1 baseline excluded ─
WEIGHTS=(0.01 0.05 0.5 1.0)
CLS_TAXA_LOSS_W="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed pretraining config (identical to fungi_its_final.sh's binary row) ──
DATASET="ITS-5M"
DATA_DIR="/project/6045013/m4safari/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

ARCH="maelm"
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
PRETRAIN_EPOCHS=15; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"

RUN_NAME="ablw_its_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_binary_w${CLS_TAXA_LOSS_W}"
CKPT_BASE="/project/6045013/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"

echo "cls-taxonomy-loss-weight: ${CLS_TAXA_LOSS_W} | Run: ${RUN_NAME}"

# ── Pretraining (skip if checkpoint already exists) ───────────────────────────
if [ -f "${CHECKPOINT_ENC}" ]; then
    echo "=== PRETRAINING SKIPPED (checkpoint exists: ${CHECKPOINT_ENC}) ==="
else
    echo "=== PRETRAINING ==="
    torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py \
        --run-name "${RUN_NAME}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE} \
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS} \
        --decoder-n-layers ${N_DEC_LAYERS} --decoder-n-heads ${N_DEC_HEADS} \
        --batch-size ${BATCH_SIZE} --lr ${LR} --weight-decay ${WD} \
        --epochs ${PRETRAIN_EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO} \
        --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT} \
        --max-norm 0.5 --separate_loss true --mixed-precision \
        --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}" \
        --checkpoint "${CHECKPOINT}" --checkpoint_maelm "${CHECKPOINT_ENC}" \
        --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS} \
        --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS} \
        --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP} \
        --use-cls-token --enable-cls-taxonomy --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}
    [ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
    echo "Pretraining done at: $(date)"
fi

[ ! -f "${CHECKPOINT_ENC}" ] && echo "ERROR: no encoder checkpoint at ${CHECKPOINT_ENC}" && exit 1

# ── KNN eval (genus-level, CLS representation -- the winning config) ─────────
OVERALL_EXIT=0
for WEIGHTS_MODE in "uniform" "softmax"; do
    WEIGHT_ARGS=(--knn-weights "${WEIGHTS_MODE}")
    [ "${WEIGHTS_MODE}" = "softmax" ] && WEIGHT_ARGS+=(--temperature 0.07)
    echo "=== KNN EVALUATION (${WEIGHTS_MODE}) ==="
    python barcodebert/knn_its_clean.py \
        --pretrained-checkpoint "${CHECKPOINT_ENC}" \
        --data-dir              "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --representation-type   cls \
        --tasks                 genus_level \
        --n-neighbors            1 3 5 7 10 15 20 25 50 \
        --metric                 cosine \
        "${WEIGHT_ARGS[@]}" \
        --run-name                "${RUN_NAME}_${WEIGHTS_MODE}" \
        --results-file             results_final/KNN_ITS_aux_weight_ablation_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN eval failed (${WEIGHTS_MODE})" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}