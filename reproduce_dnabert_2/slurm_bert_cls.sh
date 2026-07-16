#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=dnabert2_cls
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=64G
#SBATCH --time=36:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=B:USR1@3600

set -euo pipefail

SPECIES_VOCAB=${SPECIES_VOCAB:-${SHARDS_DIR:-/scratch/$USER/dnabert2_wds/shards_1.0}/species_vocab.json}
export SPECIES_VOCAB
export CONFIG_NAME=cls
export ARCHITECTURE=bert
export TRAIN_ARGS="--use-cls-token --cls-loss-weight 0.01 --species-vocab $SPECIES_VOCAB --k-classes 32 --m-per-class 2"

COMMON_LAUNCHER="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_train_and_finetune_common.sh"
if [[ ! -f "$COMMON_LAUNCHER" ]]; then
	echo "Could not find $COMMON_LAUNCHER" >&2
	echo "Submit from the reproduce_dnabert_2 directory or set SLURM_SUBMIT_DIR accordingly." >&2
	exit 1
fi

exec bash "$COMMON_LAUNCHER"
