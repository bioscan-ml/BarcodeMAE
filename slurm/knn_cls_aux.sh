#!/bin/bash
#SBATCH --job-name=knn_cls_aux
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

# Load modules
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""

VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU memory: {props.total_memory / 1024**3:.1f} GB')
"

# ── Array index → aux loss type (must match cls_aux_losses.sh) ───────────────
AUX_LOSS_TYPES=("triplet" "supcon" "ce")
AUX_LOSS_TYPE="${AUX_LOSS_TYPES[$SLURM_ARRAY_TASK_ID]}"

# ── Checkpoint path (must mirror cls_aux_losses.sh naming) ───────────────────
K_MER=6
N_LAYERS=6
N_HEADS=6
ARCH="transformer"
TAXA="genus"
K_CLASSES=8
M_PER_CLASS=4
DATASET="BIOSCAN-5M"

RUN_NAME="run_k${K_MER}_${N_LAYERS}L_${N_HEADS}H_${ARCH}_cls_aux${AUX_LOSS_TYPE}_${TAXA}_km${K_CLASSES}x${M_PER_CLASS}"
CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/model_checkpoints/${DATASET}"
CHECKPOINT="${CKPT_ROOT}/${RUN_NAME}/best_pretraining.pt"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"

mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}

echo "=========================================="
echo "Evaluating aux loss type: ${AUX_LOSS_TYPE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="

# Abort early if checkpoint doesn't exist
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

OVERALL_EXIT=0

# ── Representation types to evaluate ─────────────────────────────────────────
# cls            — CLS token at position 0 (what the aux loss was trained on)
# tokens_with_cls — mean(CLS + sequence tokens) — useful comparison baseline
REP_TYPES=("cls" "tokens_with_cls")

for REP_TYPE in "${REP_TYPES[@]}"; do
    echo ""
    echo "--- Representation type: ${REP_TYPE} ---"

    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CHECKPOINT}" \
        --dataset "${DATASET}" \
        --data-dir "${DATA_DIR}" \
        --representation_type "${REP_TYPE}" \
        --taxon genus \
        --n-neighbors 1 \
        --run-name "knn_${RUN_NAME}_${REP_TYPE}" \
        --log-wandb

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: knn_probing failed for ${RUN_NAME} / ${REP_TYPE} (exit ${EXIT_CODE})"
        OVERALL_EXIT=${EXIT_CODE}
    fi
done

echo "=========================================="
echo "Job finished at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="

exit ${OVERALL_EXIT}