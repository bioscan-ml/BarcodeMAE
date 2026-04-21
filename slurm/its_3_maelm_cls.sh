#!/bin/bash
#SBATCH --job-name=its_maelm_cls
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%1
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job ID: $SLURM_JOB_ID | Array: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="ITS-5M"
ARCH="maelm"
K_MER=6;  STRIDE=6
N_LAYERS=6;  N_HEADS=6
N_DEC_LAYERS=6;  N_DEC_HEADS=6
BATCH_SIZE=128;  LR=0.00007;  WD=0.00001
MASK_TOKEN_RATIO=1.0;  RANDOM_TOKEN_RATIO=0.0;  MASKED_LOSS_WEIGHT=0.999
K_CLASSES=8;  M_PER_CLASS=4
TAXA="genus";  TAXA_LOSS_W=0.01;  NUM_PAIRS=64

RUN_NAME="its_maelm_cls_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${N_DEC_LAYERS}DL_${N_DEC_HEADS}DH"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
CHECKPOINT_DIR="./model_checkpoints/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/${RUN_NAME}.pt"
CHECKPOINT_E="${CHECKPOINT_DIR}/${RUN_NAME}_encoder.pt"
mkdir -p "${CHECKPOINT_DIR}/finetune"

echo "Run: $RUN_NAME"

# ── Pretraining ───────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py \
    --run-name           ${RUN_NAME}          \
    --dataset            ${DATASET}           \
    --data-dir           ${DATA_DIR}          \
    --arch               ${ARCH}              \
    --k-mer              ${K_MER}             \
    --stride             ${STRIDE}            \
    --n-layers           ${N_LAYERS}          \
    --n-heads            ${N_HEADS}           \
    --decoder-n-layers   ${N_DEC_LAYERS}      \
    --decoder-n-heads    ${N_DEC_HEADS}       \
    --batch-size         ${BATCH_SIZE}        \
    --lr                 ${LR}                \
    --weight-decay       ${WD}                \
    --epochs             15                   \
    --mask-token-ratio   ${MASK_TOKEN_RATIO}  \
    --random-token-ratio ${RANDOM_TOKEN_RATIO}\
    --masked-loss-weight ${MASKED_LOSS_WEIGHT}\
    --max-norm           0.5                  \
    --separate_loss      true                 \
    --mixed-precision                         \
    --use-cls-token                           \
    --enable-cls-taxonomy                     \
    --taxonomy-level        ${TAXA}           \
    --cls-taxonomy-loss-weight ${TAXA_LOSS_W} \
    --taxonomy-max-pairs    ${NUM_PAIRS}      \
    --k-classes          ${K_CLASSES}         \
    --m-per-class        ${M_PER_CLASS}       \
    --checkpoint         ${CHECKPOINT}        \
    --checkpoint_maelm   ${CHECKPOINT_E}      \
    --save-best-model                         \
    --log-wandb

PRETRAIN_EXIT=$?
[ ${PRETRAIN_EXIT} -ne 0 ] && echo "Pretraining did not complete — skipping finetuning." && exit ${PRETRAIN_EXIT}

# ── Finetuning ────────────────────────────────────────────────────────────────
for TAXA_FT in genus species family; do
for REPR in tokens cls; do
    FINETUNE_RUN_NAME="${RUN_NAME}_ft_${TAXA_FT}_${REPR}"
    FINETUNE_CHECKPOINT="${CHECKPOINT_DIR}/finetune/${FINETUNE_RUN_NAME}.pt"
    FINETUNE_DONE_FLAG="${CHECKPOINT_DIR}/finetune/.done_${TAXA_FT}_${REPR}"

    [ -f "${FINETUNE_DONE_FLAG}" ] && echo "Finetuning (${TAXA_FT}/${REPR}) already done — skipping." && continue

    torchrun --standalone --nproc_per_node=1 barcodebert/finetuning.py \
        --run-name              ${FINETUNE_RUN_NAME}   \
        --dataset               ${DATASET}             \
        --data-dir              ${DATA_DIR}            \
        --pretrained-checkpoint ${CHECKPOINT}          \
        --checkpoint            ${FINETUNE_CHECKPOINT} \
        --taxonomic-level       ${TAXA_FT}             \
        --representation-type   ${REPR}                \
        --batch-size            64                     \
        --lr                    0.00008                \
        --weight-decay          ${WD}                  \
        --epochs                12                     \
        --mixed-precision                              \
        --save-best-model                              \
        --log-wandb

    FINETUNE_EXIT=$?
    [ ${FINETUNE_EXIT} -eq 0 ] && touch "${FINETUNE_DONE_FLAG}"
    [ ${FINETUNE_EXIT} -ne 0 ] && echo "Finetuning (${TAXA_FT}/${REPR}) failed." && exit ${FINETUNE_EXIT}
done
done

echo "Done at: $(date)" && exit 0