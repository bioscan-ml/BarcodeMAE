#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=write_shards
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem-per-cpu=16000M
#SBATCH --time=0-6:00:00
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Redirect caches away from home (quota-limited)
export HF_HOME=/scratch/m4safari/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/m4safari/cache/huggingface
export MPLCONFIGDIR=/scratch/m4safari/cache/matplotlib
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

module load python/3.11

source /scratch/m4safari/BarcodeMAE_venv/bin/activate

cd /project/6045013/m4safari/danbert2222/BarcodeMAE/reproduce_dnabert_2

echo "------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "------------------------------------------------------"

python write_shards.py