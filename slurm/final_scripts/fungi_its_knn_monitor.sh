#!/bin/bash
# ============================================================================
# Fungi ITS — Epoch-wise KNN Overfitting Monitor (maelm, genus, CLS configs)
#
# Fresh pretraining run (distinct checkpoint path from fungi_its_final.sh, so
# it always retrains from scratch rather than being skipped by the
# if-checkpoint-exists guard) with --knn-eval-every-epoch enabled: after every
# epoch, runs the full knn_its.py evaluation (whole train+test data, not a
# subsample) against that epoch's checkpoint, so you can watch train/val KNN
# accuracy epoch-by-epoch and catch overfitting as it happens.
#
# Same hyperparameters as fungi_its_final.sh (maelm, genus level, 15 epochs,
# triplet margin=0.0 with default batch-hard mining) — this is the "default"
# CLS-enabled config sweep, just with per-epoch monitoring turned on.
#
# 4 tasks (0-3):  aux task = [none, binary, triplet, ce]   (all CLS-enabled, maelm, genus)
#
# Pretraining ONLY — no finetuning step. Epoch-wise KNN logs land in
# results_final/KNN_ITS_RESULTS_epochwise.txt (appended after every epoch —
# never overwritten, so nothing is lost even if the job times out partway
# through). The last epoch's entry there is your "final" number; there is no
# separate final-eval step in this script.
# Checkpoints: main_checkpoints_final/ablations/knn_monitor/ITS-5M/
#
# Time budget: fungi_its_final.sh (no epoch-wise eval) needs 28h for 15
# pretraining epochs + finetuning; this script drops the finetuning step but
# adds a full knn_its.py pass (train gallery + all 3 fungi test sets) after
# EVERY one of those 15 epochs, which is significant added wall-clock —
# 72h budgeted accordingly. If that's not enough and the job times out
# mid-pretraining, pretraining.py resumes automatically from the last saved
# epoch checkpoint on resubmission (no results are lost — just `sbatch` this
# script again). Check your account's actual max walltime (e.g. `sacctmgr
# show assoc user=$USER format=account,maxwall`) since 72h may exceed what
# def-lila-ab allows.
#
# Submit:  sbatch fungi_its_knn_monitor.sh
# ============================================================================
#SBATCH --job-name=its_knn_monitor
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

# ── Grid (4 tasks, maelm only) ─────────────────────────────────────────────────
AUX_TASKS=("none" "binary" "triplet" "ce")
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"
ARCH="maelm"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"

# ── Fixed config (matches fungi_its_final.sh exactly) ─────────────────────────
K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; N_DEC_LAYERS=6; N_DEC_HEADS=6
BATCH_SIZE=128; LR=0.00007; WD=0.00001
MASKED_LOSS_WEIGHT=0.999; MASK_TOKEN_RATIO=1.0; RANDOM_TOKEN_RATIO=0.0
PRETRAIN_EPOCHS=15; AUX_LOSS_WEIGHT=0.1; AUX_LOSS_WARMUP=5
K_CLASSES=16; M_PER_CLASS=4; NUM_PAIRS=128; TAXA="genus"
TRIPLET_MARGIN=0.0; CLS_TAXA_LOSS_W=0.1

# ── Epoch-wise KNN probe config ────────────────────────────────────────────────
KNN_EVAL_EVERY=1
KNN_EVAL_REPR="tokens"

# ── Naming (distinct from fungi_its_final.sh so pretraining is never skipped) ──
RUN_NAME="knnmon_its_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${N_DEC_LAYERS}DL${N_DEC_HEADS}DH_${ARCH}_cls_${AUX_TASK}"

# Checkpoint root derived from DATA_DIR so it always lives in the same
# directory tree as the dataset (DATA_DIR = .../data/${DATASET}).
DATA_ROOT="$(dirname "$(dirname "${DATA_DIR}")")"
CKPT_BASE="${DATA_ROOT}/main_checkpoints_final/ablations/knn_monitor/${DATASET}/${RUN_NAME}"
CHECKPOINT="${CKPT_BASE}/checkpoint.pt"
CHECKPOINT_ENC="${CKPT_BASE}/checkpoint_encoder.pt"
mkdir -p "${CKPT_BASE}"

echo "Arch: ${ARCH} | Aux: ${AUX_TASK} | Taxa: ${TAXA} | Run: ${RUN_NAME}"

# ── Pretraining args ──────────────────────────────────────────────────────────
PRETRAIN_ARGS=(
    --run-name "${RUN_NAME}" --dataset "${DATASET}" --data-dir "${DATA_DIR}"
    --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE}
    --n-layers ${N_LAYERS} --n-heads ${N_HEADS}
    --batch-size ${BATCH_SIZE} --lr ${LR} --weight-decay ${WD}
    --epochs ${PRETRAIN_EPOCHS} --mask-token-ratio ${MASK_TOKEN_RATIO}
    --random-token-ratio ${RANDOM_TOKEN_RATIO} --masked-loss-weight ${MASKED_LOSS_WEIGHT}
    --max-norm 0.5 --separate_loss true --mixed-precision
    --save-best-model --log-wandb --wandb-project "${WANDB_PROJECT}"
    --checkpoint "${CHECKPOINT}"
    --decoder-n-layers "${N_DEC_LAYERS}" --decoder-n-heads "${N_DEC_HEADS}"
    --checkpoint_maelm "${CHECKPOINT_ENC}"
    --use-cls-token
    --taxonomy-level ${TAXA} --taxonomy-max-pairs ${NUM_PAIRS}
    --k-classes ${K_CLASSES} --m-per-class ${M_PER_CLASS}
    --aux-loss-weight ${AUX_LOSS_WEIGHT} --aux-loss-warmup-epochs ${AUX_LOSS_WARMUP}
    # --knn-eval-taxon is intentionally omitted: knn_its.py (used for ITS-5M)
    # always evaluates at species level regardless of that flag — matching the
    # final finetuning eval target, independent of the genus-level aux task.
    --knn-eval-every-epoch ${KNN_EVAL_EVERY}
    --knn-eval-representation-type ${KNN_EVAL_REPR}
)
case "${AUX_TASK}" in
    none)    : ;;  # CLS present, no aux loss — just the CLS-baseline embedding
    binary)  PRETRAIN_ARGS+=(--enable-cls-taxonomy --cls-taxonomy-loss-weight ${CLS_TAXA_LOSS_W}) ;;
    triplet) PRETRAIN_ARGS+=(--aux-loss-type triplet --triplet-margin ${TRIPLET_MARGIN}) ;;  # default mining=batch_hard
    ce)      PRETRAIN_ARGS+=(--aux-loss-type ce) ;;
esac

echo "=== PRETRAINING (with epoch-wise KNN monitoring every ${KNN_EVAL_EVERY} epoch(s)) ==="
torchrun --standalone --nproc_per_node=1 barcodebert/pretraining.py "${PRETRAIN_ARGS[@]}"
EC=$?
[ ${EC} -ne 0 ] && echo "ERROR: Pretraining failed" && exit ${EC}
echo "Pretraining done at: $(date)"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}