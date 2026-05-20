#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=maelm_24h
#SBATCH --output=%x_%A_%a.out    # %x=job-name, %A=job-ID, %a=array-index
#SBATCH --error=%x_%A_%a.err
#SBATCH --mem=64G
#SBATCH --time=24:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- Environment Setup (Executed for every job in the array) ---

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# Load modules
module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

# Activate virtual environment
source /home/pmillana/dl-dev/bin/activate

# Go to project directory
cd /home/pmillana/projects/def-lila-ab/pmillana/reproduce_dnabert_2

# Configuration
CHECKPOINT_DIR="/scratch/${USER}/maelm_checkpoints_24h"
LOG_DIR="/scratch/${USER}/maelm_logs_24h"
FINETUNE_DATA="GUE/EMP/H3"

# Ensure directories exist
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Network setup for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12346 # Changed port to avoid conflict
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# OMP threads
if [ -z "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=8
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# Print status
echo "------------------------------------------------------"
echo "Job Array Index: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "------------------------------------------------------"

# --- Training Command ---
# running main_train.py

torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         main_train.py \
         --architecture maelm \
         --batch-size 64 \
         --total-batch-size 4096 \
         --max-steps 500000 \
         --checkpoint-interval 2000 \
         --checkpoint-dir "$CHECKPOINT_DIR" \
         --log-dir "$LOG_DIR" \
         --max-lr 5e-4 \
         --warmup-steps 30000 \
         --weight-decay 0.1 \
         --finetune-data-path "$FINETUNE_DATA"
