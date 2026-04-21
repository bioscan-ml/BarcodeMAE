#!/bin/bash
#SBATCH --job-name=barcodemae_its_jumbo6
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

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

# Load modules
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

# Environment setup
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Wandb setup
export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

# GPU info
echo "=========================================="
nvidia-smi
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU memory: {props.total_memory / 1024**3:.1f} GB')
"
echo "=========================================="

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET="ITS-5M"
ARCH="maelm"

K_MER=6
STRIDE=6
N_LAYERS=6
N_HEADS=6
N_DEC_LAYERS=6
N_DEC_HEADS=6

JUMBO_M=6
SOURCE="encoder"
EXPAN_FACTOR=0

TAXA="genus"
TAXA_LOSS_W=0.01
NUM_PAIRS=64
K_CLASSES=8
M_PER_CLASS=4

BATCH_SIZE=128
LR=0.00007
WD=0.00001
MASK_TOKEN_RATIO=1.0
RANDOM_TOKEN_RATIO=0.0
MASKED_LOSS_WEIGHT=0.999

# ── Derived names ─────────────────────────────────────────────────────────────
RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${N_DEC_LAYERS}DL_${N_DEC_HEADS}DH_jumbo${JUMBO_M}_${ARCH}_${SOURCE}_${TAXA_LOSS_W}_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}_${NUM_PAIRS}_${EXPAN_FACTOR}_${BATCH_SIZE}_${LR}_${MASKED_LOSS_WEIGHT}_ITS"
CHECKPOINT_DIR="./model_checkpoints/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/${RUN_NAME}.pt"
CHECKPOINT_E="${CHECKPOINT_DIR}/${RUN_NAME}_encoder.pt"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"

mkdir -p "${CHECKPOINT_DIR}/finetune"
mkdir -p logs

echo "=========================================="
echo "Run name : $RUN_NAME"
echo "Data dir : $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# ── Pretraining ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "Starting pretraining..."
echo "=========================================="

torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py \
    --run-name           ${RUN_NAME}            \
    --dataset            ${DATASET}             \
    --data-dir           ${DATA_DIR}            \
    --arch               ${ARCH}                \
    --k-mer              ${K_MER}               \
    --stride             ${STRIDE}              \
    --n-layers           ${N_LAYERS}            \
    --n-heads            ${N_HEADS}             \
    --decoder-n-layers   ${N_DEC_LAYERS}        \
    --decoder-n-heads    ${N_DEC_HEADS}         \
    --batch-size         ${BATCH_SIZE}          \
    --lr                 ${LR}                  \
    --weight-decay       ${WD}                  \
    --epochs             15                     \
    --mask-token-ratio   ${MASK_TOKEN_RATIO}    \
    --random-token-ratio ${RANDOM_TOKEN_RATIO}  \
    --masked-loss-weight ${MASKED_LOSS_WEIGHT}  \
    --max-norm           0.5                    \
    --separate_loss      true                   \
    --mixed-precision                           \
    --k-classes          ${K_CLASSES}           \
    --m-per-class        ${M_PER_CLASS}         \
    --jumbo                                     \
    --jumbo_multiplier   ${JUMBO_M}             \
    --jumbo_source       ${SOURCE}              \
    --jumbo-mlp-expansion ${EXPAN_FACTOR}       \
    --share_jumbo_layers                        \
    --enable-taxonomy-classification            \
    --taxonomy-level     ${TAXA}                \
    --taxonomy-loss-weight ${TAXA_LOSS_W}       \
    --taxonomy-max-pairs ${NUM_PAIRS}           \
    --checkpoint         ${CHECKPOINT}          \
    --checkpoint_maelm   ${CHECKPOINT_E}        \
    --save-best-model                           \
    --log-wandb

PRETRAIN_EXIT=$?
echo "Pretraining exit code: ${PRETRAIN_EXIT}"
[ ${PRETRAIN_EXIT} -ne 0 ] && echo "Pretraining did not complete — skipping finetuning." && exit ${PRETRAIN_EXIT}

# ── Finetuning ────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Pretraining done. Starting finetuning..."
echo "=========================================="

for REPR in tokens jumbo_avg jumbo; do
    FINETUNE_RUN_NAME="${RUN_NAME}_ft_genus_${REPR}"
    FINETUNE_CHECKPOINT="${CHECKPOINT_DIR}/finetune/${FINETUNE_RUN_NAME}.pt"
    FINETUNE_DONE_FLAG="${CHECKPOINT_DIR}/finetune/.done_${REPR}"

    [ -f "${FINETUNE_DONE_FLAG}" ] && echo "Finetuning (${REPR}) already done — skipping." && continue

    echo "  Representation: ${REPR}"

    torchrun --standalone --nproc_per_node=1 barcodebert/finetuning.py \
        --run-name              ${FINETUNE_RUN_NAME}   \
        --dataset               ${DATASET}             \
        --data-dir              ${DATA_DIR}            \
        --pretrained-checkpoint ${CHECKPOINT}          \
        --checkpoint            ${FINETUNE_CHECKPOINT} \
        --taxonomic-level       genus                  \
        --representation-type   ${REPR}                \
        --batch-size            64                     \
        --lr                    0.00008                \
        --weight-decay          ${WD}                  \
        --epochs                12                     \
        --mixed-precision                              \
        --save-best-model                              \
        --log-wandb

    FINETUNE_EXIT=$?
    echo "Finetuning (${REPR}) exit code: ${FINETUNE_EXIT}"
    [ ${FINETUNE_EXIT} -eq 0 ] && touch "${FINETUNE_DONE_FLAG}"
    [ ${FINETUNE_EXIT} -ne 0 ] && echo "Finetuning (${REPR}) failed." && exit ${FINETUNE_EXIT}
done

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

exit 0