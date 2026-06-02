#!/bin/bash
set -euo pipefail

: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${ARCHITECTURE:?ARCHITECTURE is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR=${PROJECT_DIR:-$SCRIPT_DIR}
SHARDS_DIR=${SHARDS_DIR:-/scratch/${USER}/dnabert2_wds/shards_1.0}
if [[ -z "${GUE_DATA_PATH:-}" ]]; then
    _gue_default_home="/home/${USER}/projects/def-lila-ab/${USER}/reproduce_dnabert_2"
    GUE_DATA_PATH="${_gue_default_home}"
fi
SPECIES_VOCAB=${SPECIES_VOCAB:-$SHARDS_DIR/species_vocab.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/${USER}/MAE_checkpoints}
LOG_ROOT=${LOG_ROOT:-/scratch/${USER}/MAE_logs}

MAX_STEPS=${MAX_STEPS:-7908}
WARMUP_STEPS=${WARMUP_STEPS:-790}
MAX_LR=${MAX_LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MASK_RATIO=${MASK_RATIO:-0.15}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
BATCH_SIZE=${BATCH_SIZE:-64}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-4096}
NUM_WORKERS=${NUM_WORKERS:-4}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-1}

WANDB_PROJECT=${WANDB_PROJECT:-dnabert2-training}
WANDB_ENTITY=${WANDB_ENTITY:-uoguelph_mlrg}
WANDB_MODE=${WANDB_MODE:-online}

TRAIN_ARGS=${TRAIN_ARGS:-}
RUN_PREFIX=${RUN_PREFIX:-exp}

module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

PYTHON_ENV_PATH="/home/$USER/dl-dev"

if [[ ! -f "$PYTHON_ENV_PATH/bin/activate" ]]; then
    echo "Required Python environment is missing: $PYTHON_ENV_PATH/bin/activate" >&2
    exit 1
fi

source "$PYTHON_ENV_PATH/bin/activate"

if ! command -v python >/dev/null 2>&1; then
    echo "python is not available after activating $PYTHON_ENV_PATH" >&2
    exit 1
fi

# Avoid stale submit-shell GPU masks when SLURM did not provide a mapping.
# This can happen when users export CUDA_VISIBLE_DEVICES before calling sbatch.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${SLURM_STEP_GPUS:-}" && -z "${SLURM_JOB_GPUS:-}" ]]; then
    echo "[preflight] Unsetting inherited CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    unset CUDA_VISIBLE_DEVICES
fi

# Preflight: distinguish between missing GPU allocation and CPU-only PyTorch builds.
python - <<'PY'
import os
import sys
import torch

print(f"[preflight] python={sys.executable}")
print(f"[preflight] torch={torch.__version__}")
print(f"[preflight] torch.version.cuda={torch.version.cuda}")
print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"[preflight] SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', '<unset>')}")
print(f"[preflight] SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', '<unset>')}")
print(f"[preflight] cuda_available={torch.cuda.is_available()}")
print(f"[preflight] cuda_device_count={torch.cuda.device_count()}")

try:
    if torch.cuda.device_count() > 0:
        _ = torch.cuda.get_device_name(0)
except Exception as e:
    print(f"[preflight] cuda_device_query_error={e}")

if not torch.cuda.is_available():
    if torch.version.cuda is None:
        raise SystemExit(
            "[preflight] ERROR: PyTorch in /home/$USER/dl-dev appears CPU-only (torch.version.cuda is None). "
            "Reinstall a CUDA-enabled torch build in this env."
        )
    raise SystemExit(
        "[preflight] ERROR: CUDA-enabled torch is installed but no GPU is visible in this SLURM job. "
        "Check the sbatch GPU request/partition and cluster allocation."
    )
PY

# Respect SLURM-provided visibility; do not force a default GPU index.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/$USER/cache/matplotlib}
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
else
    export OMP_NUM_THREADS=8
fi

export MASTER_ADDR=$(hostname)
if [[ -z "${MASTER_PORT:-}" ]]; then
    # Use a deterministic per-job port to avoid clashes when multiple jobs
    # run concurrently on the same node. Keep a stable local fallback.
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        MASTER_PORT=$((15000 + (SLURM_JOB_ID % 40000)))
    else
        MASTER_PORT=12346
    fi
fi
export MASTER_PORT
export WORLD_SIZE=${SLURM_NTASKS:-1}
export RANK=${SLURM_PROCID:-0}
export LOCAL_RANK=${SLURM_LOCALID:-0}

RUN_ID="${RUN_PREFIX}_${ARCHITECTURE}_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
FINETUNE_TAGS="${ARCHITECTURE},${CONFIG_NAME},pretrain"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_ID"
LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

cd "$PROJECT_DIR"

echo "------------------------------------------------------"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Run ID: $RUN_ID"
echo "Architecture: $ARCHITECTURE"
echo "Configuration: $CONFIG_NAME"
echo "Project dir: $PROJECT_DIR"
echo "Python env: $PYTHON_ENV_PATH"
echo "Python executable: $(command -v python)"
echo "Shards: $SHARDS_DIR"
echo "GUE path: $GUE_DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log dir: $LOG_DIR"
echo "------------------------------------------------------"

torchrun_cmd=(
    torchrun
    --nproc_per_node=1
    --nnodes=1
    --node_rank=0
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    main_train.py
    --architecture "$ARCHITECTURE"
    --n-layers 6
    --n-heads 6
    --decoder-n-layers 6
    --decoder-n-heads 6
    --batch-size "$BATCH_SIZE"
    --total-batch-size "$TOTAL_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --warmup-steps "$WARMUP_STEPS"
    --max-lr "$MAX_LR"
    --weight-decay "$WEIGHT_DECAY"
    --mask-ratio "$MASK_RATIO"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --log-interval "$LOG_INTERVAL"
    --num-workers "$NUM_WORKERS"
    --train-shards-pattern "$SHARDS_DIR/train-*.tar"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --log-dir "$LOG_DIR"
    --wandb-project "$WANDB_PROJECT"
    --wandb-mode "$WANDB_MODE"
    --wandb-run-name "$RUN_ID"
)


# MAELM has dynamic masking/indexing patterns that are unstable with torch.compile
# (Inductor/CUDAGraph errors). Keep compile off by default for MAELM runs.
torchrun_cmd+=(--no-compile)

if [[ -n "$WANDB_ENTITY" ]]; then
    torchrun_cmd+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ -n "$TRAIN_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $TRAIN_ARGS )
    torchrun_cmd+=("${extra_args[@]}")
fi

# ---------------------------------------------------------------------------
# Time-limit guard
# SLURM sends SIGUSR1 one hour before the wall-time limit when the job script
# contains:  #SBATCH --signal=B:USR1@3600
# The handler below kills torchrun gracefully so the finetuning block below
# can still run with the latest checkpoint.
# ---------------------------------------------------------------------------
_PRETRAIN_INTERRUPTED=0
_handle_timelimit() {
    echo "[timelimit] SIGUSR1 received: ~1 h of wall time remaining." \
         "Stopping pretraining so finetuning can still run within this allocation." >&2
    if [[ -n "${_TRAIN_PID:-}" ]]; then
        kill -TERM "$_TRAIN_PID" 2>/dev/null || true
    fi
    _PRETRAIN_INTERRUPTED=1
}
trap '_handle_timelimit' USR1

"${torchrun_cmd[@]}" &
_TRAIN_PID=$!
# 'wait' can return non-zero if torchrun was killed; use || true so set -e
# doesn't abort the script before finetuning gets a chance to run.
wait "$_TRAIN_PID" || true
unset _TRAIN_PID

if (( _PRETRAIN_INTERRUPTED )); then
    echo "[timelimit] Pretraining was interrupted; attempting finetuning with latest checkpoint." >&2
fi

if [[ "$ARCHITECTURE" == "maelm" ]]; then
    FINETUNE_MODEL_TYPE=maelm
    FINETUNE_MODEL_PATH="$CHECKPOINT_DIR/latest"
else
    FINETUNE_MODEL_TYPE=bert
    FINETUNE_MODEL_PATH="$CHECKPOINT_DIR/latest_checkpoint.pt"
fi

if [[ ! -e "$FINETUNE_MODEL_PATH" ]]; then
    echo "Expected finetuning model path not found: $FINETUNE_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$GUE_DATA_PATH/GUE" ]]; then
    echo "Expected GUE directory not found at: $GUE_DATA_PATH/GUE" >&2
    echo "Set GUE_DATA_PATH to the directory that contains the GUE folder." >&2
    exit 1
fi

bash "$SCRIPT_DIR/finetune_all_maelm.sh" \
    "$GUE_DATA_PATH" \
    "$FINETUNE_MODEL_PATH" \
    "$RUN_ID" \
    "$FINETUNE_MODEL_TYPE" \
    "$WANDB_PROJECT" \
    "$WANDB_ENTITY" \
    "$WANDB_MODE" \
    "$RUN_ID" \
    "$FINETUNE_TAGS"

echo "Completed training + finetuning for $RUN_ID"
