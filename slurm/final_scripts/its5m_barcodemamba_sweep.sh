#!/bin/bash
# ============================================================================
# ITS-5M: softmax-KNN temperature sweep for BarcodeMamba+ (UNITE), the two
# released size variants (GitHub release v0.2.0 "BarcodeMamba+ weights"):
#   0 = BarcodeMamba-plus-layer2-dim384 (12.1M params after finetuning)
#   1 = BarcodeMamba-plus-layer4-dim768 (49.2M params after finetuning)
# Fills in the "BarcodeMamba" rows in tab:its_external.
#
# *** CHECKPOINT PATHS: adjust CHECKPOINT_BASE/BPE_TOKENIZER below to wherever
# you extract models_release.tar (from the v0.2.0 release) and download the
# release-level bpe_tokenizer.pkl to. Expected layout after extracting the
# tar:
#   ${CHECKPOINT_BASE}/BarcodeMamba-plus-layer2-dim384/{config.yaml,model.ckpt,overrides.yaml}
#   ${CHECKPOINT_BASE}/BarcodeMamba-plus-layer4-dim768/{config.yaml,model.ckpt,overrides.yaml}
#
# REQUIRES slurm/setup_env_barcodemamba.sh to have been run once first.
#
# Submit: sbatch slurm/final_scripts/its5m_barcodemamba_sweep.sh
# ============================================================================
#SBATCH --job-name=its5m_barcodemamba_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv_barcodemamba/bin/activate"
export WANDB_MODE=disabled
export TMPDIR="/scratch/$USER/tmp_wandb"
mkdir -p "$TMPDIR"

mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

BM_REPO="/scratch/$USER/BarcodeMamba-dev"
CHECKPOINT_BASE="/scratch/$USER/barcodemamba_checkpoints/models_release"
BPE_TOKENIZER="${CHECKPOINT_BASE}/bpe_tokenizer.pkl"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

CKPT_DIRS=(
    "${CHECKPOINT_BASE}/BarcodeMamba-plus-layer2-dim384"
    "${CHECKPOINT_BASE}/BarcodeMamba-plus-layer4-dim768"
)
TAGS=("layer2dim384" "layer4dim768")

CHECKPOINT_DIR="${CKPT_DIRS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
echo "Checkpoint: ${CHECKPOINT_DIR} | tag: ${TAG}"

echo "=== UNIFORM KNN EVALUATION ==="
python barcodebert/knn_its_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            uniform \
    --run-name                "knn_its_barcodemamba_${TAG}_uniform" \
    --results-file             results_final/KNN_ITS_external_temp_sweep_RESULTS.txt
EC0=$?; [ ${EC0} -ne 0 ] && echo "ERROR: uniform KNN eval failed for ${TAG}"

echo "=== SOFTMAX TEMPERATURE SWEEP ==="
python barcodebert/knn_its_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            softmax \
    --temperature-sweep      ${TEMPS} \
    --run-name                "knn_its_barcodemamba_${TAG}_softmax_sweep" \
    --results-file             results_final/KNN_ITS_external_temp_sweep_RESULTS.txt
EC1=$?; [ ${EC1} -ne 0 ] && echo "ERROR: temperature sweep failed for ${TAG}"

OVERALL_EXIT=0
[ ${EC0} -ne 0 ] && OVERALL_EXIT=${EC0}
[ ${EC1} -ne 0 ] && OVERALL_EXIT=${EC1}
echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}