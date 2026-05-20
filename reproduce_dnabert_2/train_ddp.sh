#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=dnabert2_train
#SBATCH --output=dnabert2_train_%j.out
#SBATCH --error=dnabert2_train_%j.err
#SBATCH --mem=64G                 # Total memory per node 
#SBATCH --time=0-23:00:00
#SBATCH --cpus-per-task=8         # CPUs per task (increased for better I/O)
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # Only one task per node for multi-GPU training
#SBATCH --exclusive               # Request exclusive access to avoid interference

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # All 4 GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Disable InfiniBand if causing issues
export PYTHONUNBUFFERED=1         # Ensure output is flushed immediately

# Load modules (Alliance Canada standard)
module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

# Activate virtual environment
source dl-dev/bin/activate

# Go to project directory
cd /home/pmillana/projects/def-lila-ab/pmillana/reproduce_dnabert_2


# Verify GPU availability
echo "Available GPUs:"
nvidia-smi


# Set environment variables for optimal performance
# Fix OMP_NUM_THREADS for multi-GPU training
if [ -z "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=8
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# WebDataset distributed training settings
export WDS_EPOCH=0
export WDS_NUM_WORKERS=4          # Match number of GPUs
export WDS_WORKER_ID=0            # Will be overridden by torchrun

# Verify GPU availability
echo "Available GPUs:"
nvidia-smi

# Set up distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Run multi-GPU distributed training
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         main_train.py \
         --architecture bert \
         --batch-size 64 \
         --total-batch-size 4096 \
         --max-steps 500000 \
         --checkpoint-interval 2000 \
         --checkpoint-dir /scratch/$USER/dnabert2_checkpoints \
         --log-dir /scratch/$USER/dnabert2_logs
