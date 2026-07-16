#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=nt_benchmark_cv
#SBATCH --output=nt_benchmark_cv_%j.out
#SBATCH --error=nt_benchmark_cv_%j.err
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

source ~/dl-dev/bin/activate

PROJECT_DIR=/home/pmillana/projects/def-lila-ab/pmillana/BarcodeMAE/reproduce_dnabert_2

# Priority: positional args > exported env vars > defaults
MODEL_NAME_OR_PATH="${1:-${MODEL_NAME_OR_PATH:-/scratch/pmillana/MAE_checkpoints/exp_maelm_auxiliary_20260527_051553/latest}}"
MODEL_TYPE="${2:-${MODEL_TYPE:-maelm}}"
RUN_NAME="${3:-${RUN_NAME:-exp_maelm_auxiliary_nt_benchmark}}"
TASK_NAMES="${4:-${TASK_NAMES:-}}"
OUTPUT_DIR="${5:-${OUTPUT_DIR:-output/nt_benchmark}}"
EXTRA_PYTHON_ARGS="${6:-${EXTRA_PYTHON_ARGS:---fp16 true}}"

cd "$PROJECT_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "MODEL_TYPE=$MODEL_TYPE"
echo "RUN_NAME=$RUN_NAME"
if [[ -n "$TASK_NAMES" ]]; then
  echo "TASK_NAMES=$TASK_NAMES"
fi
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EXTRA_PYTHON_ARGS=$EXTRA_PYTHON_ARGS"

bash ./finetune_all_nt_benchmark.sh \
  "$MODEL_NAME_OR_PATH" \
  "$MODEL_TYPE" \
  "$RUN_NAME" \
  "$TASK_NAMES" \
  "$OUTPUT_DIR" \
  "$EXTRA_PYTHON_ARGS"