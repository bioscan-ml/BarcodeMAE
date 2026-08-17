#!/bin/bash
# ============================================================================
# BIOSCAN-5M auxiliary-loss-WEIGHT ablation, for all three CLS auxiliary
# objectives (CE, Binary, Triplet), each at the same fixed architecture
# (encoder-decoder, +CLS). Sweeps the objective's own weight flag over a
# short list of values around the main run's 0.1 baseline. 0.1 itself is NOT
# included -- that checkpoint/result already exists from bioscan5m_final.sh,
# no need to retrain it.
#
# CE and Triplet share the general --aux-loss-weight flag; Binary has its own
# --cls-taxonomy-loss-weight flag (--aux-loss-weight stays fixed at the 0.1
# baseline for Binary, matching bioscan5m_final.sh's binary row -- only
# --cls-taxonomy-loss-weight varies). Flag patterns per task mirror
# bioscan5m_final.sh's case statement exactly.
#
# Unlike bioscan5m_final.sh, this is 3 SINGLE fixed configs (maelm, CLS,
# {ce,binary,triplet}) with only the weight varying -- not the full 10-config
# grid. Pretraining is immediately followed by the same genus-level KNN eval
# (uniform + softmax, k=1..50) used for the main results, so each task is
# fully self-contained.
#
# 12 array tasks (0-11): 4 weight values x 3 objectives (CE, Binary,
# Triplet). Tasks 0-3 (CE) were already run in an earlier submission (before
# this ablation was extended to all three objectives); pretraining auto-skips
# for them since their checkpoints already exist, so resubmitting the full
# 0-11 range is safe (idempotent) if you want one array to manage. To only
# submit the 8 NEW (Binary + Triplet) tasks, use: sbatch --array=4-11 <this
# script>.
#
# ── Chunked/self-chaining walltime (CHANGED) ────────────────────────────────
# Each task now requests only CHUNK_HOURS (default 6h) instead of one giant
# ~19-20h request, since a single ~26.5h request had sat 2 days with no
# resource allocation. At startup, EACH job -- before doing any expensive
# work -- pre-emptively submits its own continuation (same array index, same
# script) via `sbatch --dependency=afterany:$SLURM_JOB_ID`, so the chain
# keeps going even if this chunk is killed by SLURM mid-epoch on walltime.
# This relies on pretraining.py's existing checkpoint/resume support: it
# resumes from the last epoch saved to `${CHECKPOINT}` automatically, so
# passing the exact same --checkpoint path across chunks is all that's
# needed -- no new resume flag required.
#
# A `pretrain_done.flag` marker (written only once pretraining.py returns
# successfully, which structurally only happens once ALL epochs are trained)
# stops the chain: once present, later chunks skip straight to KNN/ZSC eval
# without resubmitting further or re-invoking pretraining.py. MAX_CHAIN_DEPTH
# (default 6, i.e. up to 6 x CHUNK_HOURS = 36h total) is a hard safety cap on
# how many chunks a single task can consume, in case something is stuck.
#
# The "skip uniform-voting KNN re-run for already-complete checkpoints"
# decision (tasks 0-3/CE, trained before this ablation existed) is now
# persisted once to a `skip_uniform.flag` marker at first touch (CHAIN_DEPTH
# 0), since checking "does checkpoint_encoder.pt exist" is no longer a valid
# proxy once chunking is in play -- it starts existing after just the first
# epoch of ANY fresh run, not only once a run is fully done.
#
# Checkpoints: main_checkpoints_final/ablations/aux_weight/BIOSCAN-5M/
# Results:     results_final/KNN_bioscan5m_aux_weight_ablation_RESULTS.txt
#              (uniform, auto-routed to the _softmax_ variant for softmax)
#
# Submit (all 12): sbatch slurm/final_scripts/bioscan5m_aux_weight_ablation.sh
# Submit (new 8 only): sbatch --array=4-11 slurm/final_scripts/bioscan5m_aux_weight_ablation.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_aux_weight_ablation
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN_DEPTH=6

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Chain depth $CHAIN_DEPTH | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/project/6045013/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

# ── Grid (12 tasks): 4 weight values x 3 objectives, 0.1 baseline excluded ──
AUX_TASKS=("ce" "ce" "ce" "ce"   "binary" "binary" "binary" "binary"   "triplet" "triplet" "triplet" "triplet")
WEIGHTS=(   0.01  0.05  0.5  1.0   0.01     0.05     0.5     1.0        0.01      0.05      0.5      1.0)
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

# ── Fixed pretraining config (identical to bioscan5m_final.sh's rows) ───────
DATASET="BIOSCAN-5M"
DATA_DIR="/project/6045013/m4safari/BarcodeMAE/data/${DATASET}"
ARCH="maelm"
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
EPOCHS=35; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"
TRIPLET_MARGIN=0.0

RUN_NAME="ablw_bioscan5m_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_${AUX_TASK}_w${WEIGHT}"
CKPT_BASE="/project/6045013/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
DONE_FLAG="${CKPT_BASE}/pretrain_done.flag"
SKIP_UNIFORM_FLAG="${CKPT_BASE}/skip_uniform.flag"
mkdir -p "${CKPT_BASE}"

echo "aux-task: ${AUX_TASK} | weight: ${WEIGHT} | Run: ${RUN_NAME}"

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
        --epochs ${EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO}
        --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT}
        --max-norm 0.5 --separate_loss true --mixed-precision
        --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}"
        --checkpoint "${CHECKPOINT}" --checkpoint_maelm "${CHECKPOINT_ENC}"
        --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS}
        --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
        --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    )
    case "${AUX_TASK}" in
        ce)      PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type ce --aux-loss-weight ${WEIGHT}) ;;
        binary)  PRETRAIN_ARGS+=(--use-cls-token --enable-cls-taxonomy --cls-taxonomy-loss-weight ${WEIGHT}) ;;
        triplet) PRETRAIN_ARGS+=(--use-cls-token --aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN} --aux-loss-weight ${WEIGHT}) ;;
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
# skip_uniform.flag means this checkpoint was already fully trained (at the
# old T=0.07 default) before this ablation existed -- its uniform-voting KNN
# results are unaffected by the T change, so skip re-running it. Only softmax
# (at the new T=0.02) and ZSC (never run before) are (re)run in that case.
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
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CHECKPOINT_ENC}" \
        --dataset                "${DATASET}" --data-dir "${DATA_DIR}" \
        --representation_type    cls --taxon genus \
        --n-neighbors             1 3 5 7 10 15 20 25 50 \
        --metric                  cosine \
        "${WEIGHT_ARGS[@]}" \
        --run-name                 "${RUN_NAME}_${WEIGHTS_MODE}" \
        --results-file              results_final/KNN_bioscan5m_aux_weight_ablation_RESULTS.txt \
        --wandb-project "${WANDB_PROJECT}" --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN eval failed (${WEIGHTS_MODE})" && OVERALL_EXIT=${EC}
done

# ── ZSC eval (genus-level, CLS representation, open-world BIN reconstruction) ─
echo "=== ZSC EVALUATION ==="
python barcodebert/zsc_evaluation_v2.py \
    --pretrained-checkpoint "${CHECKPOINT_ENC}" \
    --dataset                "${DATASET}" --data-dir "${DATA_DIR}" \
    --representation_type    cls --taxon genus \
    --n-neighbors             15 --metric cosine \
    --run-name                 "zsc_${RUN_NAME}" \
    --results-file              results_final/ZSC_bioscan5m_aux_weight_ablation_RESULTS.txt \
    --wandb-project "${WANDB_PROJECT}" --log-wandb
EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: ZSC eval failed" && OVERALL_EXIT=${EC}

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}