#!/bin/bash
set -euo pipefail

data_path=${1:?Usage: $0 <data_path> <model_path> <run_name> [model_type] [wandb_project] [wandb_entity] [wandb_mode] [wandb_group] [extra_tags]}
MODEL_NAME_OR_PATH=${2:?Usage: $0 <data_path> <model_path> <run_name> [model_type] [wandb_project] [wandb_entity] [wandb_mode] [wandb_group] [extra_tags]}
RUN_NAME=${3:?Usage: $0 <data_path> <model_path> <run_name> [model_type] [wandb_project] [wandb_entity] [wandb_mode] [wandb_group] [extra_tags]}
MODEL_TYPE=${4:-maelm}
WANDB_PROJECT=${5:-dnabert2-training}
WANDB_ENTITY=${6:-}
WANDB_MODE=${7:-online}
WANDB_GROUP=${8:-}
EXTRA_TAGS=${9:-}
lr=3e-5

echo "data_path=${data_path}"
echo "model_path=${MODEL_NAME_OR_PATH}"
echo "run_name=${RUN_NAME}"
echo "model_type=${MODEL_TYPE}"

BASE_TAGS="finetuning,${MODEL_TYPE}"
if [[ -n "$EXTRA_TAGS" ]]; then
    BASE_TAGS="${BASE_TAGS},${EXTRA_TAGS}"
fi

run_finetune() {
    local dataset_path="$1"
    local run_suffix="$2"
    local model_max_length="$3"
    local train_bs="$4"
    local eval_bs="$5"
    local epochs="$6"
    local save_steps="$7"
    local eval_steps="$8"
    local warmup_steps="$9"
    local output_dir="${10}"
    local extra_args="${11:-}"

    local cmd=(
        python finetune.py
        --model_name_or_path "$MODEL_NAME_OR_PATH"
        --model_type "$MODEL_TYPE"
        --data_path "$dataset_path"
        --kmer -1
        --run_name "$run_suffix"
        --model_max_length "$model_max_length"
        --per_device_train_batch_size "$train_bs"
        --per_device_eval_batch_size "$eval_bs"
        --gradient_accumulation_steps 1
        --learning_rate "$lr"
        --num_train_epochs "$epochs"
        --fp16
        --save_steps "$save_steps"
        --output_dir "$output_dir"
        --evaluation_strategy steps
        --eval_steps "$eval_steps"
        --warmup_steps "$warmup_steps"
        --logging_steps 100000
        --overwrite_output_dir True
        --log_level info
        --find_unused_parameters False
        --wandb_project "$WANDB_PROJECT"
        --wandb_mode "$WANDB_MODE"
        --wandb_tags "$BASE_TAGS"
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
        cmd+=(--wandb_entity "$WANDB_ENTITY")
    fi

    if [[ -n "$WANDB_GROUP" ]]; then
        cmd+=(--wandb_group "$WANDB_GROUP")
    fi

    if [[ -n "$extra_args" ]]; then
        # shellcheck disable=SC2206
        cmd+=( $extra_args )
    fi

    "${cmd[@]}"
}

for seed in 42
do
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    do
        run_finetune \
            "$data_path/GUE/EMP/$data" \
            "${RUN_NAME}/EMP_${data}_seed${seed}" \
            128 8 16 3 200 200 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}/EMP_${data}_seed${seed}"
    done

    for data in prom_core_all prom_core_notata
    do
        run_finetune \
            "$data_path/GUE/prom/$data" \
            "${RUN_NAME}_prom_${data}_seed${seed}" \
            20 8 16 4 400 400 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_prom_${data}_seed${seed}"
    done

    for data in prom_core_tata
    do
        run_finetune \
            "$data_path/GUE/prom/$data" \
            "${RUN_NAME}_prom_${data}_seed${seed}" \
            20 8 16 10 200 200 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_prom_${data}_seed${seed}"
    done

    for data in prom_300_all prom_300_notata
    do
        run_finetune \
            "$data_path/GUE/prom/$data" \
            "${RUN_NAME}_prom_${data}_seed${seed}" \
            70 8 16 4 400 400 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_prom_${data}_seed${seed}"
    done

    for data in prom_300_tata
    do
        run_finetune \
            "$data_path/GUE/prom/$data" \
            "${RUN_NAME}_prom_${data}_seed${seed}" \
            70 8 16 10 200 200 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_prom_${data}_seed${seed}"
    done

    for data in reconstructed
    do
        run_finetune \
            "$data_path/GUE/splice/$data" \
            "${RUN_NAME}_splice_${data}_seed${seed}" \
            80 8 16 5 200 200 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_splice_${data}_seed${seed}"
    done

    for data in covid
    do
        run_finetune \
            "$data_path/GUE/virus/$data" \
            "${RUN_NAME}_virus_${data}_seed${seed}" \
            256 32 32 8 200 200 50 \
            "output/${MODEL_TYPE}/${RUN_NAME}_virus_${data}_seed${seed}"
    done

    for data in 0 1 2 3 4
    do
        run_finetune \
            "$data_path/GUE/mouse/$data" \
            "${RUN_NAME}_mouse_${data}_seed${seed}" \
            30 8 64 5 200 200 30 \
            "output/${MODEL_TYPE}/${RUN_NAME}_mouse_${data}_seed${seed}" \
            "--max_steps 1000"
    done

    for data in 0 1 2 3 4
    do
        run_finetune \
            "$data_path/GUE/tf/$data" \
            "${RUN_NAME}_tf_${data}_seed${seed}" \
            30 8 64 3 200 200 30 \
            "output/${MODEL_TYPE}/${RUN_NAME}_tf_${data}_seed${seed}"
    done
done
