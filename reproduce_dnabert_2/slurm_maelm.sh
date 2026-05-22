#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=maelm_auxiliary
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=64G
#SBATCH --time=24:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

export CONFIG_NAME=auxiliary
export ARCHITECTURE=maelm
export TRAIN_ARGS=""

COMMON_LAUNCHER="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_train_and_finetune_common.sh"
if [[ ! -f "$COMMON_LAUNCHER" ]]; then
	echo "Could not find $COMMON_LAUNCHER" >&2
	echo "Submit from the reproduce_dnabert_2 directory or set SLURM_SUBMIT_DIR accordingly." >&2
	exit 1
fi

bash "$COMMON_LAUNCHER"
