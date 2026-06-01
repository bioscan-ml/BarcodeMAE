#!/bin/bash
#SBATCH --job-name=barcodeMAE_cls_aux_transformer
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-2%3
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

echo "Loaded modules:"
module list

# Environment setup
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# Activate virtual environment
VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Wandb setup
export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

# Verify GPU setup
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

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

# ── Array index → aux loss type ───────────────────────────────────────────────
AUX_LOSS_TYPES=("triplet" "supcon" "ce")
AUX_LOSS_TYPE="${AUX_LOSS_TYPES[$SLURM_ARRAY_TASK_ID]}"

echo "Aux loss type: $AUX_LOSS_TYPE"

# ── Base configuration ────────────────────────────────────────────────────────
K_MER=6
STRIDE=6
N_LAYERS=6
N_HEADS=6
MASKED_LOSS_WEIGHT=0.999
WD=0.00001
RANDOM_TOKEN_RATIO=0.0
MASK_TOKEN_RATIO=1.0
ARCH="transformer"
DATASET="BIOSCAN-5M"
K_CLASSES=8
M_PER_CLASS=4
NUM_PAIRS=64
BATCH_SIZE=128
TAXA="genus"
LR=0.00007

# ── Aux loss hyperparameters ──────────────────────────────────────────────────
AUX_LOSS_WEIGHT=0.1
TRIPLET_MARGIN=0.3
SUPCON_TEMP=0.15         # raised from 0.07; softer distribution helps sparse embedding spaces
AUX_LOSS_WARMUP=5        # ramp aux weight from 0→full over this many epochs

# ── Names and paths ───────────────────────────────────────────────────────────
RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${ARCH}_cls_aux${AUX_LOSS_TYPE}_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}"
CHECKPOINT_DIR="./model_checkpoints/${DATASET}/aux_4/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pt"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}
mkdir -p logs

echo "=========================================="
echo "Configuration:"
echo "  Run name:        $RUN_NAME"
echo "  Aux loss type:   $AUX_LOSS_TYPE"
echo "  Aux loss weight: $AUX_LOSS_WEIGHT"
echo "  Triplet margin:  $TRIPLET_MARGIN  (ignored for supcon/ce)"
echo "  SupCon temp:     $SUPCON_TEMP     (ignored for triplet/ce)"
echo "  Encoder:         ${N_LAYERS}L × ${N_HEADS}H  (transformer, no decoder)"
echo "  k×m sampler:     k=${K_CLASSES}, m=${M_PER_CLASS}"
echo "=========================================="

torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py \
    --run-name "${RUN_NAME}" \
    --dataset "${DATASET}" \
    --data-dir /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET} \
    --arch "${ARCH}" \
    --k-mer ${K_MER} \
    --stride ${STRIDE} \
    --n-layers ${N_LAYERS} \
    --n-heads ${N_HEADS} \
    --separate_loss true \
    --masked-loss-weight ${MASKED_LOSS_WEIGHT} \
    --mask-token-ratio ${MASK_TOKEN_RATIO} \
    --random-token-ratio ${RANDOM_TOKEN_RATIO} \
    --batch-size ${BATCH_SIZE} \
    --k-classes ${K_CLASSES} \
    --m-per-class ${M_PER_CLASS} \
    --lr ${LR} \
    --weight-decay ${WD} \
    --max-norm 0.5 \
    --epochs 35 \
    --mixed-precision \
    --save-best-model \
    --log-wandb \
    --checkpoint "${CHECKPOINT}" \
    --use-cls-token \
    --taxonomy-level ${TAXA} \
    --taxonomy-max-pairs ${NUM_PAIRS} \
    --aux-loss-type ${AUX_LOSS_TYPE} \
    --aux-loss-weight ${AUX_LOSS_WEIGHT} \
    --triplet-margin ${TRIPLET_MARGIN} \
    --supcon-temperature ${SUPCON_TEMP} \
    --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="