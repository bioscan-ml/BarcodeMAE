#!/bin/bash
#SBATCH --job-name=barcodemae_h100x2
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # torchrun manages GPU processes internally
#SBATCH --gres=gpu:h100:2             # 2x H100 GPUs
#SBATCH --cpus-per-task=8             # 4 CPUs per GPU * 2 GPUs
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --array=0-30%1
#SBATCH --output=logs/h100_single_%A_%a.out
#SBATCH --error=logs/h100_single_%A_%a.err

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
export WANDB_DIR="/scratch/$USER/wandb"
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

# Configuration
K_MER=6
STRIDE=6
MASKED_LOSS_WEIGHT=0.999
N_LAYERS=6
N_HEADS=6
N_DEC_HEADS=6
N_DEC_LAYERS=6
SOURCE="encoder"
WD=0.00001
RANDOM_TOKEN_RATIO=0.0
MASK_TOKEN_RATIO=1.0
ARCH="maelm"
DATASET="BIOSCAN-5M"
JUMBO_M=6
K_CLASSES=4
M_PER_CLASS=2
RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H__${N_DEC_LAYERS}DL_${N_DEC_HEADS}DH_jumbo${JUMBO_M}_${ARCH}_${SOURCE}_0.01_nopooling_family_km${K_CLASSES}x${M_PER_CLASS}"
CHECKPOINT_DIR="./model_checkpoints/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/${RUN_NAME}.pt"
CHECKPOINT_E="${CHECKPOINT_DIR}/${RUN_NAME}_encoder.pt"
mkdir -p ${CHECKPOINT_DIR}
mkdir -p logs

echo "=========================================="
echo "Configuration:"
echo "  Run name: $RUN_NAME"
echo "  GPUs: 2"
echo "  Batch size per GPU: 64 (k=${K_CLASSES}, m=${M_PER_CLASS})"
echo "=========================================="

# Run distributed training (torchrun manages both GPU processes)
torchrun --standalone --nproc_per_node=2 barcodebert/pretraining.py \
    --run-name ${RUN_NAME} \
    --k-mer ${K_MER} \
    --stride ${STRIDE} \
    --data-dir /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET} \
    --separate_loss true \
    --checkpoint ${CHECKPOINT} \
    --masked-loss-weight ${MASKED_LOSS_WEIGHT} \
    --log-wandb \
    --n-layers ${N_LAYERS} \
    --n-heads ${N_HEADS} \
    --mask-token-ratio ${MASK_TOKEN_RATIO} \
    --random-token-ratio ${RANDOM_TOKEN_RATIO} \
    --save-best-model \
    --epochs 35 \
    --dataset ${DATASET} \
    --weight-decay ${WD} \
    --arch ${ARCH} \
    --batch-size 64 \
    --jumbo \
    --jumbo_multiplier ${JUMBO_M} \
    --enable-taxonomy-classification \
    --taxonomy-level family \
    --taxonomy-loss-weight 0.01 \
    --k-classes ${K_CLASSES} \
    --m-per-class ${M_PER_CLASS} \
    --share_jumbo_layers \
    --checkpoint_maelm ${CHECKPOINT_E} \
    --jumbo_source ${SOURCE} \
    --decoder-n-heads ${N_DEC_HEADS} \
    --decoder-n-layers ${N_DEC_LAYERS} \
    #--pool_jumbo_for_taxonomy \

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE