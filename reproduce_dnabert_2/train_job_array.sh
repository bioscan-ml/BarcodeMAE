#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=dnabert2_train
#SBATCH --output=%x_%A_%a.out    # %x=job-name, %A=job-ID, %a=array-index
#SBATCH --error=%x_%A_%a.err
#SBATCH --mem=64G
#SBATCH --time=23:55:00           # Just under 24h limit
#SBATCH --cpus-per-task=16        # Increased for dataloader workers
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-7%1             # 7 jobs, run 1 at a time (approx 7 days total)

# --- Environment Setup (Executed for every job in the array) ---

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
CHECKPOINT_DIR="/scratch/${USER}/dnabert2_checkpoints"
LOG_DIR="/scratch/${USER}/dnabert2_logs"
FINETUNE_DATA="GUE/splice/reconstructed"

# Ensure directories exist
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Network setup for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
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
echo "Checking for existing checkpoint in $CHECKPOINT_DIR..."
ls -l $CHECKPOINT_DIR/latest_checkpoint.pt 2>/dev/null || echo "No checkpoint found, starting fresh."
echo "------------------------------------------------------"

# --- Training Command ---
# train.py is configured to automatically look for 'latest_checkpoint.pt' 
# inside --checkpoint-dir. If found, it resumes. If not, it starts from step 0.

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py \
         --batch-size 64 \
         --total-batch-size 4096 \
         --max-steps 500000 \
         --checkpoint-interval 2000 \
         --checkpoint-dir "$CHECKPOINT_DIR" \
         --log-dir "$LOG_DIR" \
         --finetune-data-path "$FINETUNE_DATA"

# Note: If the training finishes before the array is done, subsequent jobs
# will simply load the finished model, see it's at max_steps, and exit quickly.
