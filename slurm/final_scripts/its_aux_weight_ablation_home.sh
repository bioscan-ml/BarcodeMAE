#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) auxiliary-loss-WEIGHT ablation -- SAME 12 tasks as
# its_aux_weight_ablation.sh, but for a DIFFERENT cluster whose project
# storage is only reachable via the ~/projects/def-lila-ab/ symlink form
# (no /project/6045013/ numeric mount there). Directory convention here
# matches fungi_its_knn_monitor.sh's DATA_DIR/CKPT_BASE/WANDB_DIR pattern
# exactly. Account, modules, and venv path are otherwise identical to
# its_aux_weight_ablation.sh.
#
# For CE and Triplet share the general --aux-loss-weight flag; Binary has its
# own --cls-taxonomy-loss-weight flag (--aux-loss-weight stays fixed at the
# 0.1 baseline for Binary). Flag patterns per task mirror
# fungi_its_final.sh's case statement exactly.
#
# Pretraining is immediately followed by the same leakage-free genus-level
# KNN eval (uniform + softmax, k=1..50) used for the main results -- no
# finetuning step, since the paper's KNN numbers use the raw pretrained
# encoder (checkpoint_encoder.pt) directly.
#
# REQUIRES its_export_tasks.sh already run on this cluster (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv under DATA_DIR below).
#
# 12 array tasks (0-11): 4 weight values x 3 objectives (Binary, Triplet,
# CE). 0.1 baseline excluded -- that's fungi_its_final.sh's row.
#
# ── Chunked/self-chaining walltime ──────────────────────────────────────────
# Same pattern as its_aux_weight_ablation.sh: each task requests only 3h;
# at startup, EACH job -- before doing any expensive work -- pre-emptively
# submits its own continuation (same array index, same script) via
# `sbatch --dependency=afterany:$SLURM_JOB_ID`, so the chain keeps going even
# if this chunk is killed by SLURM mid-epoch on walltime. Relies on
# pretraining.py's existing checkpoint/resume support: it resumes from the
# last epoch saved to `${CHECKPOINT}` automatically, so passing the exact
# same --checkpoint path across chunks is all that's needed.
#
# A `pretrain_done.flag` marker (written only once pretraining.py returns
# successfully, which structurally only happens once ALL epochs are trained)
# stops the chain: once present, later chunks skip straight to KNN eval.
# MAX_CHAIN_DEPTH (default 12, i.e. up to 12 x 3h = 36h total) is a hard
# safety cap on how many chunks a single task can consume.
#
# This is a fresh cluster, so none of the 12 tasks should have a pre-existing
# checkpoint from an earlier submission -- the skip_uniform.flag machinery is
# kept anyway (harmless no-op) purely so this script stays a drop-in mirror
# of its_aux_weight_ablation.sh.
#
# Checkpoints: <DATA_ROOT>/main_checkpoints_final/ablations/aux_weight/ITS-5M/
# Results:     results_final/KNN_ITS_aux_weight_ablation_RESULTS.txt
#              (uniform, auto-routed to the _softmax_ variant for softmax)
#
# Submit (all 12): sbatch slurm/final_scripts/its_aux_weight_ablation_home.sh
# Submit (new 8 only): sbatch --array=4-11 slurm/final_scripts/its_aux_weight_ablation_home.sh
# ============================================================================
#SBATCH --job-name=its_aux_weight_ablation_home
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN_DEPTH=12

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Chain depth $CHAIN_DEPTH | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

# ── Grid (12 tasks): 4 weight values x 3 objectives, 0.1 baseline excluded ──
AUX_TASKS=("binary" "binary" "binary" "binary"   "triplet" "triplet" "triplet" "triplet"   "ce" "ce" "ce" "ce")
WEIGHTS=(   0.01     0.05     0.5     1.0          0.01      0.05      0.5      1.0          0.01  0.05  0.5  1.0)
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed pretraining config (identical to fungi_its_final.sh's rows) ───────
DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

# Checkpoint/wandb root derived from DATA_DIR (mirrors fungi_its_knn_monitor.sh):
# DATA_DIR = <DATA_ROOT>/data/${DATASET}, so two dirname's up gets DATA_ROOT.
DATA_ROOT="$(dirname "$(dirname "${DATA_DIR}")")"

export WANDB_MODE=offline
export WANDB_DIR="${DATA_ROOT}/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

ARCH="maelm"
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
PRETRAIN_EPOCHS=15; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"
TRIPLET_MARGIN=0.0

RUN_NAME="ablw_its_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_${AUX_TASK}_w${WEIGHT}"
CKPT_BASE="${DATA_ROOT}/main_checkpoints_final/ablations/aux_weight/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
DONE_FLAG="${CKPT_BASE}/pretrain_done.flag"
SKIP_UNIFORM_FLAG="${CKPT_BASE}/skip_uniform.flag"
mkdir -p "${CKPT_BASE}"

echo "aux-task: ${AUX_TASK} | weight: ${WEIGHT} | Run: ${RUN_NAME}"
echo "Data root: ${DATA_ROOT}"

# ── Persist the "was this checkpoint already fully trained before this
# ablation existed" decision exactly once, at first touch (chain depth 0).
# Later chunks must NOT re-derive this from checkpoint_encoder.pt's mere
# existence, since that file now also appears after just one partial epoch.
if [ "${CHAIN_DEPTH}" -eq 0 ] && [ ! -f "${SKIP_UNIFORM_FLAG}" ] && [ ! -f "${DONE_FLAG}" ]; then
    if [ -f "${CHECKPOINT_ENC}" ]; then
        echo "Checkpoint already existed at first touch -- marking skip_uniform.flag"
        touch "${SKIP_UNIFORM_FLAG}"
    fi
fi

# ── Pre-emptively queue the next chunk before doing any expensive work, so
# the chain survives even if THIS chunk is walltime-killed mid-epoch. A chunk
# that starts and finds pretrain_done.flag already set skips this (nothing
# left to chain).
if [ ! -f "${DONE_FLAG}" ]; then
    if [ "${CHAIN_DEPTH}" -lt "${MAX_CHAIN_DEPTH}" ]; then
        NEXT_DEPTH=$((CHAIN_DEPTH + 1))
        NEXT_JOB=$(sbatch --parsable \
            --dependency=afterany:${SLURM_JOB_ID} \
            --array=${SLURM_ARRAY_TASK_ID} \
            --export=ALL,CHAIN_DEPTH=${NEXT_DEPTH} \
            "$0")
        echo "Queued continuation job ${NEXT_JOB} (chain depth ${NEXT_DEPTH}) in case this chunk times out"
    else
        echo "WARNING: reached MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH} without a pretrain_done.flag -- not queuing further chunks. Investigate before resubmitting manually."
    fi
fi

# ── Pretraining (skip entirely if already fully done) ───────────────────────
if [ -f "${DONE_FLAG}" ]; then
    echo "=== PRETRAINING ALREADY COMPLETE (${DONE_FLAG} exists) -- skipping ==="
else
    echo "=== PRETRAINING (this chunk) ==="
    PRETRAIN_ARGS=(
        --run-name "${RUN_NAME}" --dataset "${DATASET}" --data-dir "${DATA_DIR}"
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE}
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS}
        --decoder-n-layers ${N_DEC_LAYERS} --decoder-n-heads ${N_DEC_HEADS}
        --batch-size ${BATCH_SIZE} --lr ${LR} --weight-decay ${WD}
        --epochs ${PRETRAIN_EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO}
        --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT}
        --max-norm 0.5 --separate_loss true --mixed-precision
        --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}"
        --checkpoint "${CHECKPOINT}" --checkpoint_maelm "${CHECKPOINT_ENC}"
        --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS}
        --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
        --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    )
    case "${AUX_TASK}" in
        binary)  PRETRAIN_ARGS+=(--use-cls-token --enable-cls-taxonomy --cls-taxonomy-loss-weight ${WEIGHT}) ;;
        triplet) PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN} --aux-loss-weight ${WEIGHT}) ;;
        ce)      PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type ce --aux-loss-weight ${WEIGHT}) ;;
    esac
    # NOTE: --checkpoint is the same path across every chunk, so pretraining.py's
    # built-in resume logic picks up from the last epoch saved by a previous
    # chunk automatically. If this chunk is walltime-killed mid-epoch, this line
    # never returns and the rest of the script never runs -- that's fine, the
    # continuation job queued above will retry from the last completed epoch.
    torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
    [ $? -ne 0 ] && echo "ERROR: Pretraining failed" && exit 1
    # Reaching this line proves pretraining.py's epoch loop is fully done
    # (it cannot return 0 with epochs remaining -- see pretraining.py main()).
    touch "${DONE_FLAG}"
    echo "Pretraining fully complete at: $(date)"
fi

[ ! -f "${CHECKPOINT_ENC}" ] && echo "ERROR: no encoder checkpoint at ${CHECKPOINT_ENC}" && exit 1

# ── KNN eval (genus-level, CLS representation -- the winning config) ─────────
OVERALL_EXIT=0
if [ -f "${SKIP_UNIFORM_FLAG}" ]; then
    EVAL_MODES=("softmax")
    echo "skip_uniform.flag set: skipping uniform KNN re-run (unaffected by T change)"
else
    EVAL_MODES=("uniform" "softmax")
fi
for WEIGHTS_MODE in "${EVAL_MODES[@]}"; do
    WEIGHT_ARGS=(--knn-weights "${WEIGHTS_MODE}")
    [ "${WEIGHTS_MODE}" = "softmax" ] && WEIGHT_ARGS+=(--temperature 0.02)
    echo "=== KNN EVALUATION (${WEIGHTS_MODE}) ==="
    python barcodebert/knn_its_clean.py \
        --pretrained-checkpoint "${CHECKPOINT_ENC}" \
        --data-dir              "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --representation-type   cls \
        --tasks                 genus_level \
        --n-neighbors            1 3 5 7 10 15 20 25 50 \
        --metric                 cosine \
        "${WEIGHT_ARGS[@]}" \
        --run-name                "${RUN_NAME}_${WEIGHTS_MODE}" \
        --results-file             results_final/KNN_ITS_aux_weight_ablation_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN eval failed (${WEIGHTS_MODE})" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}