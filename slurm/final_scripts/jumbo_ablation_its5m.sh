#!/bin/bash
# ============================================================================
# Jumbo CLS token ablation — UNITE+INSD (ITS-5M), encoder-decoder (maelm),
# best auxiliary objective for this dataset (Binary / pairwise same-genus
# classification, matching enc-dec+CLS+Binary+CLS from the main sweep), with
# the CLS token replaced by a Jumbo token of width J*D processed by a
# dedicated wide FFN (Fuller et al. 2025).
#
# Binary is applied via the DEDICATED Jumbo taxonomy classifier
# (--enable-taxonomy-classification), NOT --aux-loss-type: this is a
# different code path from the BIOSCAN-5M jumbo script (which uses CE via
# --aux-loss-type ce on mean-pooled jumbo tokens), because Binary's
# pairwise same-taxon head only exists as the dedicated JumboTaxonomyClassifier
# (see jumbo_taxonomy_classifier.py), mirroring the standard (non-jumbo)
# --enable-cls-taxonomy path used for the main CLS+Binary sweep.
#
# 6 array tasks (0-5): Jumbo multiplier J x MLP expansion factor E
#   Task | J (jumbo_multiplier) | E (jumbo-mlp-expansion)
#   -----|----------------------|------------------------
#     0  |          1           |   1x
#     1  |          3           |   1x
#     2  |          6           |   1x
#     3  |          1           |   2x
#     4  |          3           |   2x
#     5  |          6           |   2x
#
# Eval per task: leakage-free genus-level KNN (uniform + softmax voting,
# k=1,3,5,7,10,15,20,25,50, cosine metric, T=0.07) on Yeast + Filamentous,
# via knn_its_clean.py, across reps: tokens | jumbo (flattened J*D, the CLS
# analogue) | all_tokens (jumbo + sequence tokens, the Tokens+CLS analogue).
# knn_its_clean.py's extract_representations() was extended to support these
# jumbo representation types (previously BIOSCAN-5M-only, via
# representations_from_df in datasets.py) -- see knn_its_clean.py.
# No ZSC: BIN reconstruction is a BIOSCAN-5M-specific task, not run for
# UNITE+INSD anywhere else in this project either.
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv).
#
# Results: results_final/KNN_ITS_jumbo.txt (uniform, auto-routed to
# results_final/KNN_softmax_ITS_jumbo.txt for softmax by knn_results_path()).
# Checkpoints: main_checkpoints_final/ablations/jumbo/ITS-5M/
#
# Submit: sbatch slurm/final_scripts/jumbo_ablation_its5m.sh
# ============================================================================
#SBATCH --job-name=jumbo_abl_its5m
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
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
J_VALS=(  1 3 6 1 3 6 )
MLP_VALS=(1 1 1 2 2 2 )

J="${J_VALS[$SLURM_ARRAY_TASK_ID]}"
E="${MLP_VALS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed config (matches fungi_its_final.sh maelm/cls_binary exactly) ────────
DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=15; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5; TAXA="genus"
CLS_TAXA_LOSS_W=0.1; TEMPERATURE=0.07

# ── Naming ────────────────────────────────────────────────────────────────────
RUN_NAME="jumbo_j${J}_mlp${E}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_maelm_binary"

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/jumbo/${DATASET}/${RUN_NAME}"
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
    --enable-taxonomy-classification --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}
    --taxonomy-level ${TAXA}
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
        python barcodebert/knn_its_clean.py \
            --pretrained-checkpoint "${EVAL_CKPT}" \
            --data-dir              "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
            --representation-type   "${REP_TYPE}" \
            --n-neighbors           1 3 5 7 10 15 20 25 50 \
            --metric                cosine \
            "${WEIGHT_ARGS[@]}" \
            --run-name              "knn_${RUN_NAME}_${REP_TYPE}_${WEIGHTS}" \
            --results-file          results_final/KNN_ITS_jumbo.txt \
            --wandb-project         "${WANDB_PROJECT}" \
            --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for ${REP_TYPE}/${WEIGHTS}" && OVERALL_EXIT=${EC}
    done
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}