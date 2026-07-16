#!/bin/bash
set -euo pipefail

MODEL_NAME_OR_PATH=${1:?Usage: $0 <model_name_or_path> <model_type> <run_name> [task_names] [output_dir] [extra_python_args]}
MODEL_TYPE=${2:?Usage: $0 <model_name_or_path> <model_type> <run_name> [task_names] [output_dir] [extra_python_args]}
RUN_NAME=${3:?Usage: $0 <model_name_or_path> <model_type> <run_name> [task_names] [output_dir] [extra_python_args]}
TASK_NAMES=${4:-}
OUTPUT_DIR=${5:-output/nt_benchmark}
EXTRA_PYTHON_ARGS=${6:-}

cmd=(
    python nt_benchmark_cv.py
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --model_type "$MODEL_TYPE"
    --run_name "$RUN_NAME"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$TASK_NAMES" ]]; then
    cmd+=(--task_names "$TASK_NAMES")
fi

if [[ -n "$EXTRA_PYTHON_ARGS" ]]; then
    # shellcheck disable=SC2206
    cmd+=( $EXTRA_PYTHON_ARGS )
fi

echo "model_name_or_path=${MODEL_NAME_OR_PATH}"
echo "model_type=${MODEL_TYPE}"
echo "run_name=${RUN_NAME}"
if [[ -n "$TASK_NAMES" ]]; then
    echo "task_names=${TASK_NAMES}"
fi
echo "output_dir=${OUTPUT_DIR}"

"${cmd[@]}"
