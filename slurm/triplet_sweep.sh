#!/bin/bash
#SBATCH --job-name=triplet_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-4%5
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

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

# ── Sweep grid (task → hyperparameters) ──────────────────────────────────────
#
#  task 0: k=8,  margin=0.5  — original k, larger margin only
#  task 1: k=16, margin=0.3  — more classes only
#  task 2: k=16, margin=0.5  — more classes + larger margin
#  task 3: k=8,  margin=0.0  — no margin / hardest (softplus(d_pos - d_neg))
#  task 4: k=16, margin=0.0  — more classes + no margin / hardest
#
K_VALUES=(8 16 16 8 16)
M_VALUES=(4  4  4  4  4)
MARGIN_VALUES=(0.5 0.3 0.5 0.0 0.0)

K_CLASSES="${K_VALUES[$SLURM_ARRAY_TASK_ID]}"
M_PER_CLASS="${M_VALUES[$SLURM_ARRAY_TASK_ID]}"
TRIPLET_MARGIN="${MARGIN_VALUES[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed config ──────────────────────────────────────────────────────────────
K_MER=6
STRIDE=6
N_LAYERS=6
N_HEADS=6
N_DEC_LAYERS=6
N_DEC_HEADS=6
MASKED_LOSS_WEIGHT=0.999
WD=0.00001
RANDOM_TOKEN_RATIO=0.0
MASK_TOKEN_RATIO=1.0
ARCH="maelm"
DATASET="BIOSCAN-5M"
TAXA="genus"
BATCH_SIZE=128
LR=0.00007
AUX_LOSS_WEIGHT=0.1
AUX_LOSS_WARMUP=5

# ── Names and paths ───────────────────────────────────────────────────────────
# Encode k, m, and margin in the run name so each task has a unique checkpoint
MARGIN_STR=$(echo "${TRIPLET_MARGIN}" | tr '.' 'p')   # 0.5 → 0p5
RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${N_DEC_LAYERS}DL_${N_DEC_HEADS}DH_${ARCH}_cls_auxtriplet_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}_mg${MARGIN_STR}"
CHECKPOINT_DIR="./model_checkpoints/${DATASET}/aux_sweep_triplet/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pt"
CHECKPOINT_ENCODER="${CHECKPOINT_DIR}/checkpoint_encoder.pt"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}
mkdir -p logs

echo "=========================================="
echo "Configuration:"
echo "  Run name:      $RUN_NAME"
echo "  k classes:     $K_CLASSES"
echo "  m per class:   $M_PER_CLASS"
echo "  k*m (labeled): $((K_CLASSES * M_PER_CLASS))  fill: $((BATCH_SIZE - K_CLASSES * M_PER_CLASS))"
echo "  margin:        $TRIPLET_MARGIN"
echo "  aux weight:    $AUX_LOSS_WEIGHT  (warmup: ${AUX_LOSS_WARMUP} epochs)"
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
    --decoder-n-layers ${N_DEC_LAYERS} \
    --decoder-n-heads ${N_DEC_HEADS} \
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
    --checkpoint_maelm "${CHECKPOINT_ENCODER}" \
    --use-cls-token \
    --taxonomy-level ${TAXA} \
    --aux-loss-type triplet \
    --aux-loss-weight ${AUX_LOSS_WEIGHT} \
    --triplet-margin ${TRIPLET_MARGIN} \
    --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="