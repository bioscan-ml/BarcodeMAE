#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=maelm_auxiliary
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=64G
#SBATCH --time=24:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

export CONFIG_NAME=auxiliary
export ARCHITECTURE=maelm
export TRAIN_ARGS=""

bash "$(dirname "$0")/slurm_train_and_finetune_common.sh"
