#!/bin/bash
# ============================================================================
# Jumbo CLS token ablation — BIOSCAN-5M, encoder-decoder (maelm), best
# auxiliary objective (CE / direct genus cross-entropy), matching the winning
# configuration from bioscan5m_final.sh task 4 (final_..._maelm_cls_ce), but
# with the CLS token replaced by a Jumbo token of width J*D processed by a
# dedicated wide FFN (Fuller et al. 2025). CE is applied to the mean-pooled
# jumbo tokens via --aux-loss-type ce (see pretraining.py: out.jumbo_tokens
# .mean(dim=1) -> crossentropy_taxonomy_loss), the direct jumbo analogue of
# the standard CLS+CE path.
#
# Counterpart to jumbo_ablation_bioscan5m.sh, identical except it targets
# /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE (no "_final") for
# DATA_DIR/WANDB_DIR/CKPT_BASE, matching where the aux-weight-ablation
# checkpoints actually live on this cluster.
#
# 3 array tasks (0-2): Jumbo multiplier J at MLP expansion factor 1x -- the
# only configs actually reported in the appendix (MLP=2x is not currently
# reported, left incomplete). alpha=1.0 to match the current main-text CE
# checkpoint's weight.
#   Task | J (jumbo_multiplier)
#   -----|----------------------
#     0  |          1
#     1  |          3
#     2  |          6
#
# J=1 is NOT equivalent to the standard CLS run: even at J=1 the jumbo token
# is routed through its own dedicated wide FFN (Jumbo MLP) after each
# attention layer, unlike the plain CLS token which shares the per-token FFN.
#
# Eval per task: KNN (uniform + softmax voting, k=1,3,5,7,10,15,20,25,50,
# cosine metric, T=0.07) and ZSC (k=15), across reps: tokens | jumbo
# (flattened J*D, the CLS analogue) | all_tokens (jumbo + sequence tokens
# averaged, the Tokens+CLS analogue). jumbo_avg (mean-pooled jumbo tokens) is
# NOT evaluated separately here to keep the grid aligned 1:1 with the main
# tokens/cls/tokens_with_cls sweep.
#
# Results: results_final/KNN_jumbo_RESULTS_final.txt (uniform, auto-routed to
# KNN_softmax_jumbo_RESULTS_final.txt for softmax by knn_results_path()) and
# results_final/ZSC_jumbo_RESULTS_final.txt.
# Checkpoints: main_checkpoints_final/ablations/jumbo/BIOSCAN-5M/
#
# Submit: sbatch slurm/final_scripts/jumbo_ablation_bioscan5m_altpath.sh
# ============================================================================
#SBATCH --job-name=jumbo_abl_bioscan5m
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
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
J_VALS=(1 3 6)

J="${J_VALS[$SLURM_ARRAY_TASK_ID]}"
E=1

# ── Fixed config (matches bioscan5m_final.sh maelm/cls_ce exactly) ────────────
DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=35; AUX_LOSS_WEIGHT=1.0; AUX_LOSS_WARMUP=5; TAXA="genus"
TEMPERATURE=0.07

# ── Naming ────────────────────────────────────────────────────────────────────
RUN_NAME="jumbo_j${J}_mlp${E}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_maelm_ce_aux${AUX_LOSS_WEIGHT}"

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/jumbo/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"

echo "J: ${J} | MLP expansion: ${E} | Run: ${RUN_NAME}"

# ── Pretraining args ──────────────────────────────────────────────────────────
PRETRAIN_ARGS=(
    --run-name "${RUN_NAME}" --dataset "${DATASET}" --data-dir "${DATA_DIR}"
    --arch maelm --k-mer ${K_MER} --stride ${STRIDE}
    --n-layers ${N_LAYERS} --n-heads ${N_HEADS}
    --decoder-n-layers "${N_DEC_LAYERS}" --decoder-n-heads "${N_DEC_HEADS}"
    --batch-size ${BATCH_SIZE} --lr ${LR} --weight-decay ${WD}
    --epochs ${EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO}
    --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT}
    --max-norm 0.5 --separate_loss true --mixed-precision
    --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}"
    --checkpoint "${CHECKPOINT}" --checkpoint_maelm "${CHECKPOINT_ENC}"
    --jumbo --jumbo_multiplier ${J} --jumbo-mlp-expansion ${E}
    --aux-loss-type ce --taxonomy-level ${TAXA}
    --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
)

# ── Pretraining ───────────────────────────────────────────────────────────────
echo "=== PRETRAINING ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
[ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
echo "Pretraining done at: $(date)"

# ── Eval checkpoint ───────────────────────────────────────────────────────────
EVAL_CKPT="${CHECKPOINT_ENC}"
[ ! -f "${EVAL_CKPT}" ] && EVAL_CKPT="${CKPT_BASE}/best_model.pt"
[ ! -f "${EVAL_CKPT}" ] && echo "ERROR: no eval checkpoint found" && exit 1

REP_TYPES=("tokens" "jumbo" "all_tokens")
OVERALL_EXIT=0

# ── KNN (uniform + softmax voting, k = 1,3,5,7,10,15,20,25,50) ────────────────
echo "=== KNN EVALUATION ==="
for REP_TYPE in "${REP_TYPES[@]}"; do
    for WEIGHTS in "uniform" "softmax"; do
        echo "--- KNN repr=${REP_TYPE} weights=${WEIGHTS} ---"
        WEIGHT_ARGS=(--knn-weights "${WEIGHTS}")
        [ "${WEIGHTS}" = "softmax" ] && WEIGHT_ARGS+=(--temperature ${TEMPERATURE})
        python barcodebert/knn_probing.py \
            --pretrained-checkpoint "${EVAL_CKPT}" \
            --dataset               "${DATASET}" \
            --data-dir              "${DATA_DIR}" \
            --representation_type   "${REP_TYPE}" \
            --taxon                 genus \
            --n-neighbors           1 3 5 7 10 15 20 25 50 \
            --metric                cosine \
            "${WEIGHT_ARGS[@]}" \
            --run-name              "knn_${RUN_NAME}_${REP_TYPE}_${WEIGHTS}" \
            --results-file          results_final/KNN_jumbo_RESULTS_final.txt \
            --wandb-project         "${WANDB_PROJECT}" \
            --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for ${REP_TYPE}/${WEIGHTS}" && OVERALL_EXIT=${EC}
    done
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
        --results-file          results_final/ZSC_jumbo_RESULTS_final.txt \
        --wandb-project         "${WANDB_PROJECT}" \
        --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: ZSC failed for ${REP_TYPE}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}