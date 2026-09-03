#!/bin/bash
# ============================================================================
# Ablation: Aux Task Taxonomy Level Sweep
#
# Tests whether training the CLS at a coarser level (family) changes
# downstream representation quality vs genus (used in main experiments).
# BIN level is covered separately by ablation_taxonomy_level_bin.sh; "order"
# is not run.
#
# 6 aux configs × 1 taxonomy level (family) = 6 tasks
#   task 0: maelm       + binary
#   task 1: maelm       + triplet
#   task 2: maelm       + ce
#   task 3: transformer + binary
#   task 4: transformer + triplet
#   task 5: transformer + ce
#
# BIOSCAN-5M: KNN/ZSC evaluation is run at genus level (same as main
# experiments) so results are directly comparable.
# ITS-5M: pretraining only, no eval — species-level knn_its.py eval is
# superseded by the leakage-free knn_its_clean.py pipeline (see
# its_knn_clean_eval.sh), run separately once these checkpoints exist.
#
# Time budget: 24h. Should have generous headroom for both datasets: BIOSCAN
# has no finetuning step (pretrain -> KNN/ZSC only, so less than
# fungi_its_final.sh's 28h despite more epochs), and ITS-5M has neither
# finetuning nor eval at all now (pretrain only).
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M ablation_taxonomy_level.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     ablation_taxonomy_level.sh
# ============================================================================
#SBATCH --job-name=abl_taxa_level
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final

WANDB_PROJECT="barcodemae_cls"

# ── Grid (6 tasks) ────────────────────────────────────────────────────────────
ARCHS=(
    "maelm"        # task 0: binary
    "maelm"        # task 1: triplet
    "maelm"        # task 2: ce
    "transformer"  # task 3: binary
    "transformer"  # task 4: triplet
    "transformer"  # task 5: ce
)
AUX_TASKS=(
    "binary"
    "triplet"
    "ce"
    "binary"
    "triplet"
    "ce"
)
TAXA_LEVELS=(
    "family"
    "family"
    "family"
    "family"
    "family"
    "family"
)

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
TAXA="${TAXA_LEVELS[$SLURM_ARRAY_TASK_ID]}"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
    EPOCHS=15
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
    EPOCHS=35
fi

# ── Fixed config ──────────────────────────────────────────────────────────────
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
# Matches the main experiments' per-dataset alpha (Experimental.tex: alpha=1.0
# for BIOSCAN-5M, alpha=0.1 for UNITE+INSD) so this ablation is comparable to
# the genus-level main results instead of using a mismatched fixed weight.
if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    AUX_LOSS_WEIGHT=1.0
else
    AUX_LOSS_WEIGHT=0.1
fi
AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128
TRIPLET_MARGIN=0.0; CLS_TAXA_LOSS_W=0.1

# ── Naming ────────────────────────────────────────────────────────────────────
if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="abl_taxa${TAXA}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_${AUX_TASK}_aux${AUX_LOSS_WEIGHT}"
else
    RUN_NAME="abl_taxa${TAXA}_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_cls_${AUX_TASK}_aux${AUX_LOSS_WEIGHT}"
fi

# Checkpoint root derived from DATA_DIR so it always lives in the same
# directory tree as the dataset (DATA_DIR = .../data/${DATASET}).
DATA_ROOT="$(dirname "$(dirname "${DATA_DIR}")")"
CKPT_BASE="${DATA_ROOT}/main_checkpoints_final/ablations/taxonomy_level/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

echo "Arch: ${ARCH} | AuxTask: ${AUX_TASK} | AuxTaxa: ${TAXA} | Dataset: ${DATASET}"
echo "Run: ${RUN_NAME}"

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
    --use-cls-token
    --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS}
    --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
    --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
)
[ "${ARCH}" = "maelm" ] && PRETRAIN_ARGS+=(
    --decoder-n-layers "${N_DEC_LAYERS}" --decoder-n-heads "${N_DEC_HEADS}"
    --checkpoint_maelm "${CHECKPOINT_ENC}"
)
case "${AUX_TASK}" in
    binary)  PRETRAIN_ARGS+=(--enable-cls-taxonomy --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}) ;;
    triplet) PRETRAIN_ARGS+=(--aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN}) ;;
    ce)      PRETRAIN_ARGS+=(--aux-loss-type ce) ;;
esac

echo "=== PRETRAINING (aux task trained at ${TAXA} level) ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
[ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
echo "Pretraining done at: $(date)"

[ "${ARCH}" = "maelm" ] && EVAL_CKPT="${CHECKPOINT_ENC}" || EVAL_CKPT="${CHECKPOINT}"
[ ! -f "${EVAL_CKPT}" ] && echo "ERROR: no eval checkpoint" && exit 1

OVERALL_EXIT=0

if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    # Evaluate at genus level (same as main experiments → directly comparable)
    for REP_TYPE in "tokens" "cls" "tokens_with_cls"; do
        python barcodebert/knn_probing.py \
            --pretrained-checkpoint "${EVAL_CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 1 3 5 7 \
            --run-name "knn_${RUN_NAME}_${REP_TYPE}" \
            --results-file results_final/KNN_RESULTS_final_abl_taxalevel.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}

        python barcodebert/zsc_evaluation_v2.py \
            --pretrained-checkpoint "${EVAL_CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
            --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 15 --metric cosine \
            --run-name "zsc_${RUN_NAME}_${REP_TYPE}" \
            --results-file results_final/ZSC_RESULTS_final_abl_taxalevel.txt \
            --wandb-project "${WANDB_PROJECT}" --log-wandb
        EC=$?; [ ${EC} -ne 0 ] && OVERALL_EXIT=${EC}
    done
fi
# ITS-5M: pretraining only, no eval here — knn_its.py's species-level eval is
# superseded by the leakage-free knn_its_clean.py pipeline (see
# its_knn_clean_eval.sh), which will cover these checkpoints once trained.

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}