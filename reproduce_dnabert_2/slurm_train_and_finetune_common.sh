#!/bin/bash
set -euo pipefail

: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${ARCHITECTURE:?ARCHITECTURE is required}"

PROJECT_DIR=${PROJECT_DIR:-/home/pmillana/projects/def-lila-ab/pmillana/BarcodeMAE/reproduce_dnabert_2}
SHARDS_DIR=${SHARDS_DIR:-/scratch/${USER}/dnabert2_wds/shards_1.0}
GUE_DATA_PATH=${GUE_DATA_PATH:-/home/pmillana/projects/def-lila-ab/pmillana/reproduce_dnabert_2/}
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
LOG_INTERVAL=${LOG_INTERVAL:-100}

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

source /home/pmillana/dl-dev/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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
export MASTER_PORT=${MASTER_PORT:-12346}
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
echo "Shards: $SHARDS_DIR"
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

"${torchrun_cmd[@]}"

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

bash finetune_all_maelm.sh \
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
