#!/bin/bash
#SBATCH --job-name=jumbo_aux
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-5%6
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

# Pretraining on BIOSCAN-5M with Jumbo (J=6, mlp_expansion=2) + best aux loss configs.
# Aux embedding: jumbo_tokens.mean(dim=1)  (jumbo_avg)
#
#  task 0: MAELM       + triplet  (k=16, margin=0.0)
#  task 1: MAELM       + supcon   (k=8,  τ=0.15)
#  task 2: MAELM       + ce       (k=8)
#  task 3: transformer + triplet  (k=16, margin=0.0)
#  task 4: transformer + supcon   (k=8,  τ=0.15)
#  task 5: transformer + ce       (k=8)

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""

VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU memory: {props.total_memory / 1024**3:.1f} GB')
"

# ── Sweep grid ────────────────────────────────────────────────────────────────
ARCHS=(      "maelm"   "maelm"  "maelm"  "transformer" "transformer" "transformer")
AUX_TYPES=(  "triplet" "supcon" "ce"     "triplet"     "supcon"      "ce"         )
K_VALUES=(   16        8        8        16            8             8            )
M_VALUES=(   4         4        4        4             4             4            )
MARGINS=(    0.0       0.3      0.3      0.0           0.3           0.3         )
TEMPS=(      0.07      0.15     0.07     0.07          0.15          0.07        )

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
AUX_TYPE="${AUX_TYPES[$SLURM_ARRAY_TASK_ID]}"
K_CLASSES="${K_VALUES[$SLURM_ARRAY_TASK_ID]}"
M_PER_CLASS="${M_VALUES[$SLURM_ARRAY_TASK_ID]}"
TRIPLET_MARGIN="${MARGINS[$SLURM_ARRAY_TASK_ID]}"
SUPCON_TEMP="${TEMPS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed config ──────────────────────────────────────────────────────────────
K_MER=6
STRIDE=6
N_LAYERS=6
N_HEADS=6
N_DEC_LAYERS=6
N_DEC_HEADS=6
JUMBO_J=6          # number of jumbo CLS tokens
JUMBO_EXPAN=2      # MLP expansion factor for jumbo
JUMBO_SOURCE="encoder"
MASKED_LOSS_WEIGHT=0.999
WD=0.00001
RANDOM_TOKEN_RATIO=0.0
MASK_TOKEN_RATIO=1.0
DATASET="BIOSCAN-5M"
NUM_PAIRS=64
BATCH_SIZE=128
TAXA="genus"
LR=0.00007
AUX_LOSS_WEIGHT=0.1
AUX_LOSS_WARMUP=5

# ── Names and paths ───────────────────────────────────────────────────────────
MARGIN_STR=$(echo "${TRIPLET_MARGIN}" | tr '.' 'p')

if [ "${ARCH}" = "maelm" ]; then
    RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${N_DEC_LAYERS}DL_${N_DEC_HEADS}DH_${ARCH}_jumbo${JUMBO_J}x${JUMBO_EXPAN}_aux${AUX_TYPE}_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}_mg${MARGIN_STR}"
else
    RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${ARCH}_jumbo${JUMBO_J}x${JUMBO_EXPAN}_aux${AUX_TYPE}_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}_mg${MARGIN_STR}"
fi

CHECKPOINT_DIR="./model_checkpoints/${DATASET}/aux_jumbo/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pt"
CHECKPOINT_ENCODER="${CHECKPOINT_DIR}/checkpoint_encoder.pt"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}

echo "=========================================="
echo "Configuration:"
echo "  Arch:          ${ARCH}"
echo "  Aux loss:      ${AUX_TYPE}  (weight=${AUX_LOSS_WEIGHT}, warmup=${AUX_LOSS_WARMUP})"
echo "  Triplet margin: ${TRIPLET_MARGIN}  SupCon τ: ${SUPCON_TEMP}"
echo "  k×m sampler:   k=${K_CLASSES}, m=${M_PER_CLASS}"
echo "  Jumbo:         J=${JUMBO_J}, expansion=${JUMBO_EXPAN}, source=${JUMBO_SOURCE}"
echo "  Run name:      ${RUN_NAME}"
echo "=========================================="

# ── Build pretraining args ────────────────────────────────────────────────────
PRETRAIN_ARGS=(
    --run-name             "${RUN_NAME}"
    --dataset              "${DATASET}"
    --data-dir             "/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
    --arch                 "${ARCH}"
    --k-mer                "${K_MER}"
    --stride               "${STRIDE}"
    --n-layers             "${N_LAYERS}"
    --n-heads              "${N_HEADS}"
    --batch-size           "${BATCH_SIZE}"
    --lr                   "${LR}"
    --weight-decay         "${WD}"
    --epochs               35
    --mask-token-ratio     "${MASK_TOKEN_RATIO}"
    --random-token-ratio   "${RANDOM_TOKEN_RATIO}"
    --masked-loss-weight   "${MASKED_LOSS_WEIGHT}"
    --max-norm             0.5
    --separate_loss        true
    --mixed-precision
    --jumbo
    --jumbo_multiplier     "${JUMBO_J}"
    --jumbo_source         "${JUMBO_SOURCE}"
    --jumbo-mlp-expansion  "${JUMBO_EXPAN}"
    --share_jumbo_layers
    --k-classes            "${K_CLASSES}"
    --m-per-class          "${M_PER_CLASS}"
    --taxonomy-level       "${TAXA}"
    --taxonomy-max-pairs   "${NUM_PAIRS}"
    --aux-loss-type        "${AUX_TYPE}"
    --aux-loss-weight      "${AUX_LOSS_WEIGHT}"
    --triplet-margin       "${TRIPLET_MARGIN}"
    --supcon-temperature   "${SUPCON_TEMP}"
    --aux-loss-warmup-epochs "${AUX_LOSS_WARMUP}"
    --checkpoint           "${CHECKPOINT}"
    --save-best-model
    --log-wandb
)

if [ "${ARCH}" = "maelm" ]; then
    PRETRAIN_ARGS+=(
        --decoder-n-layers "${N_DEC_LAYERS}"
        --decoder-n-heads  "${N_DEC_HEADS}"
        --checkpoint_maelm "${CHECKPOINT_ENCODER}"
    )
fi

torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="