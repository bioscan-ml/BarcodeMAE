#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <architecture: maelm|bert> <config: auxiliary|cls|cls_jumbo>"
    exit 1
fi

ARCHITECTURE="$1"
CONFIG_NAME="$2"

if [[ "$ARCHITECTURE" != "maelm" && "$ARCHITECTURE" != "bert" ]]; then
    echo "Invalid architecture: $ARCHITECTURE"
    exit 1
fi

if [[ "$CONFIG_NAME" != "auxiliary" && "$CONFIG_NAME" != "cls" && "$CONFIG_NAME" != "cls_jumbo" ]]; then
    echo "Invalid config: $CONFIG_NAME"
    exit 1
fi

PROJECT_DIR=${PROJECT_DIR:-/home/loan/Nextcloud/CodeRepos/BarcodeMAE/reproduce_dnabert_2}
SHARDS_DIR=${SHARDS_DIR:-/scratch/${USER}/dnabert2_wds/shards_1.0}
GUE_DATA_PATH=${GUE_DATA_PATH:-/scratch/${USER}/dnabert2_wds}
SPECIES_VOCAB=${SPECIES_VOCAB:-$SHARDS_DIR/species_vocab.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/${USER}/MAE_checkpoints}
LOG_ROOT=${LOG_ROOT:-/scratch/${USER}/MAE_logs}

MAX_STEPS=${MAX_STEPS:-7908}
WARMUP_STEPS=${WARMUP_STEPS:-600}
MAX_LR=${MAX_LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MASK_RATIO=${MASK_RATIO:-0.15}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
BATCH_SIZE=${BATCH_SIZE:-64}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-4096}
NUM_WORKERS=${NUM_WORKERS:-4}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-100}
LOCAL_NPROC=${LOCAL_NPROC:-1}

WANDB_PROJECT=${WANDB_PROJECT:-dnabert2-training}
WANDB_ENTITY=${WANDB_ENTITY:-uoguelph_mlrg}
WANDB_MODE=${WANDB_MODE:-online}
RUN_PREFIX=${RUN_PREFIX:-localtest}

ENABLE_FINETUNE=${ENABLE_FINETUNE:-1}
TRAIN_ARGS=${TRAIN_ARGS:-}

if [[ -n "${VENV_PATH:-}" ]]; then
    source "$VENV_PATH/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/$USER/cache/matplotlib}
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

case "$CONFIG_NAME" in
    auxiliary)
        CONFIG_TRAIN_ARGS=""
        ;;
    cls)
        CONFIG_TRAIN_ARGS="--use-cls-token --cls-loss-weight 0.01 --species-vocab $SPECIES_VOCAB --k-classes 32 --m-per-class 2"
        ;;
    cls_jumbo)
        CONFIG_TRAIN_ARGS="--jumbo --jumbo-multiplier 6 --jumbo-mlp-expansion 2 --share-jumbo-layers --cls-loss-weight 0.01 --species-vocab $SPECIES_VOCAB --k-classes 32 --m-per-class 2 --no-compile"
        ;;
esac

RUN_ID="${RUN_PREFIX}_${ARCHITECTURE}_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
FINETUNE_TAGS="${ARCHITECTURE},${CONFIG_NAME},pretrain"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_ID"
LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

cd "$PROJECT_DIR"

echo "------------------------------------------------------"
echo "Date: $(date)"
echo "Run ID: $RUN_ID"
echo "Architecture: $ARCHITECTURE"
echo "Configuration: $CONFIG_NAME"
echo "Project dir: $PROJECT_DIR"
echo "Shards: $SHARDS_DIR"
echo "GUE path: $GUE_DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log dir: $LOG_DIR"
echo "ENABLE_FINETUNE: $ENABLE_FINETUNE"
echo "------------------------------------------------------"

torchrun_cmd=(
    torchrun
    --nproc_per_node="$LOCAL_NPROC"
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port=12346
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

if [[ -n "$WANDB_ENTITY" ]]; then
    torchrun_cmd+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ -n "$CONFIG_TRAIN_ARGS" ]]; then
    # shellcheck disable=SC2206
    conf_args=( $CONFIG_TRAIN_ARGS )
    torchrun_cmd+=("${conf_args[@]}")
fi

if [[ -n "$TRAIN_ARGS" ]]; then
    # shellcheck disable=SC2206
    user_args=( $TRAIN_ARGS )
    torchrun_cmd+=("${user_args[@]}")
fi

"${torchrun_cmd[@]}"

if [[ "$ENABLE_FINETUNE" != "1" ]]; then
    echo "Skipping finetuning because ENABLE_FINETUNE=$ENABLE_FINETUNE"
    exit 0
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

echo "Completed local training + finetuning for $RUN_ID"
