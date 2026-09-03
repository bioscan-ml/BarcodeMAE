#!/bin/bash
# ============================================================================
# ITS-5M Jumbo CLS token ablation -- RE-EVAL ONLY for the "jumbo" (CLS-analogue)
# and "all_tokens" (Tok+CLS-analogue) representations, against the two
# already-trained checkpoints (J=3 and J=6, MLP expansion 1x). Counterpart to
# jumbo_ablation_its5m.sh, which already produced complete "tokens"
# representation results for these checkpoints but failed for "jumbo" (no
# traceback in the logs -- consistent with a silent OOM kill from the wider
# J*D-dim gallery embeddings) and never reached "all_tokens" at all (the
# whole job was killed, most likely by its walltime, before getting there).
#
# Narval counterpart to jumbo_ablation_its5m_reeval.sh, submitted here
# because fir is under maintenance. Only difference from the fir version:
# gres switched h100 -> a100 (narval's def-lila-ab allocation this session
# has consistently been a100, not h100). Both checkpoints rsync'd over from
# fir into main_checkpoints_final/ablations/jumbo/ITS-5M/ complete
# (best_pretraining.pt, checkpoint.pt, checkpoint_encoder.pt all present).
#
# 8 array tasks (0-7): 2 checkpoints x 2 reprs (jumbo, all_tokens) x
# 2 voting modes (uniform, softmax).
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv) -- same requirement as the
# original script.
#
# Results: results_final/KNN_ITS_jumbo.txt (same file as the original
# script, so these rows land alongside the existing "tokens" results).
#
# Submit everything:      sbatch slurm/final_scripts/jumbo_ablation_its5m_reeval_narval.sh
# Submit specific tasks:  sbatch --array=0,1 slurm/final_scripts/jumbo_ablation_its5m_reeval_narval.sh
# ============================================================================
#SBATCH --job-name=jumbo_its5m_reeval
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/jumbo/${DATASET}"

# ── Grid (8 tasks): 2 checkpoints x 2 reprs x 2 voting modes ─────────────────
RUN_NAMES=(
    "jumbo_j3_mlp1_k6_6L6H_6DL6DH_maelm_binary"
    "jumbo_j6_mlp1_k6_6L6H_6DL6DH_maelm_binary"
)
REPRS=("jumbo" "all_tokens")
WEIGHTS_LIST=("uniform" "softmax")

TOTAL=0
GRID_RUN_NAME=(); GRID_REPR=(); GRID_WEIGHTS=()
for RN in "${RUN_NAMES[@]}"; do
    for R in "${REPRS[@]}"; do
        for W in "${WEIGHTS_LIST[@]}"; do
            GRID_RUN_NAME+=("${RN}"); GRID_REPR+=("${R}"); GRID_WEIGHTS+=("${W}")
            TOTAL=$((TOTAL + 1))
        done
    done
done
echo "Grid has ${TOTAL} entries (expected 8)"

RUN_NAME="${GRID_RUN_NAME[$SLURM_ARRAY_TASK_ID]}"
REP_TYPE="${GRID_REPR[$SLURM_ARRAY_TASK_ID]}"
WEIGHTS="${GRID_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

CKPT="${CKPT_ROOT}/${RUN_NAME}/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

echo "Run: ${RUN_NAME} | Repr: ${REP_TYPE} | Weights: ${WEIGHTS} | Ckpt: ${CKPT}"

WEIGHT_ARGS=(--knn-weights "${WEIGHTS}")
[ "${WEIGHTS}" = "softmax" ] && WEIGHT_ARGS+=(--temperature 0.07)

python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir              "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type   "${REP_TYPE}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    "${WEIGHT_ARGS[@]}" \
    --run-name               "knn_${RUN_NAME}_${REP_TYPE}_${WEIGHTS}" \
    --results-file            results_final/KNN_ITS_jumbo.txt \
    --wandb-project "${WANDB_PROJECT}" --log-wandb
EC=$?

[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for ${REP_TYPE}/${WEIGHTS}"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}