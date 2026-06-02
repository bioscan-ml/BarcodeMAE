#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=dnabert2_finetune_24h
#SBATCH --output=finetune_24h_%j.out
#SBATCH --error=finetune_24h_%j.err
#SBATCH --mem=16G
#SBATCH --time=0-5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

# Load modules (Alliance Canada standard)
module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

# Activate virtual environment
source /home/pmillana/dl-dev/bin/activate

PROJECT_DIR=/home/pmillana/projects/def-lila-ab/pmillana/BarcodeMAE/reproduce_dnabert_2
GUE_DATA_PATH=/home/pmillana/projects/def-lila-ab/pmillana/reproduce_dnabert_2
MODEL_PATH=/scratch/pmillana/MAE_checkpoints/exp_maelm_auxiliary_20260527_051553/latest
RUN_NAME=exp_maelm_auxiliary_retry_finetune
MODEL_TYPE=maelm

cd "$PROJECT_DIR"

bash ./finetune_all_maelm.sh \
  "$GUE_DATA_PATH" \
  "$MODEL_PATH" \
  "$RUN_NAME" \
  "$MODEL_TYPE"
