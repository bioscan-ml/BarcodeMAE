#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=maelm_jumbo_tax
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=64G
#SBATCH --time=24:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- Environment ---

export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# Redirect HF and matplotlib caches away from home (quota-limited)
export HF_HOME=/scratch/m4safari/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/m4safari/cache/huggingface
export MPLCONFIGDIR=/scratch/m4safari/cache/matplotlib
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# Load modules
module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

# Activate virtual environment
source /scratch/m4safari/BarcodeMAE_venv/bin/activate

# Go to project directory
cd /project/6045013/m4safari/danbert2222/BarcodeMAE/reproduce_dnabert_2

# OMP threads
if [ -z "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=8
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# --- Paths ---

SHARDS_DIR="/scratch/m4safari/dnabert2_wds/shards_0.01"
SPECIES_VOCAB="$SHARDS_DIR/species_vocab.json"
CHECKPOINT_DIR="/scratch/m4safari/checkpoints/maelm_jumbo_6l6h"
LOG_DIR="/scratch/m4safari/logs/maelm_jumbo_6l6h"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Network setup for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12346
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print status
echo "------------------------------------------------------"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Shards:   $SHARDS_DIR"
echo "------------------------------------------------------"

# --- Training ---
# Architecture : MAELM + Jumbo CLS + taxonomy head
# Encoder/Decoder : 6 layers, 6 attention heads
# Jumbo           : J=6 CLS tokens, MLP expansion x2, shared weights
# Pairs           : k=32 species x m=2 samples -> 32 pos + 32 neg per batch
# Batch           : 128 sequences (64 labeled + 64 fill)

torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         main_train.py \
         --architecture maelm \
         \
         --n-layers 6 \
         --n-heads 6 \
         --decoder-n-layers 6 \
         --decoder-n-heads 6 \
         \
         --jumbo \
         --jumbo-multiplier 6 \
         --jumbo-mlp-expansion 2 \
         --share-jumbo-layers \
         --cls-loss-weight 0.01 \
         --species-vocab "$SPECIES_VOCAB" \
         \
         --k-classes 32 \
         --m-per-class 2 \
         \
         --batch-size 128 \
         --total-batch-size 4096 \
         --max-steps 500000 \
         --warmup-steps 30000 \
         --max-lr 5e-4 \
         --weight-decay 0.1 \
         --mask-ratio 0.15 \
         --max-seq-length 224 \
         \
         --checkpoint-interval 2000 \
         --log-interval 100 \
         --compile-mode reduce-overhead \
         --num-workers 4 \
         --train-shards-pattern "$SHARDS_DIR/train-*.tar" \
         --checkpoint-dir "$CHECKPOINT_DIR" \
         --log-dir "$LOG_DIR"