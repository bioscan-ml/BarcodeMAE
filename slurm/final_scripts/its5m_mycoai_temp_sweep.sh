#!/bin/bash
# ============================================================================
# ITS-5M MycoAI-BERT / MycoAI-CNN: softmax-KNN temperature sweep over their
# latent_space() embeddings (knn_its_mycoai.py), same treatment as the other
# external baselines. Uniform-KNN results already exist -- this only adds
# the new softmax column.
#
# *** CHECKPOINT PATHS: adjust MYCOAI_BERT_CKPT/MYCOAI_CNN_CKPT below if
# these aren't where your checkpoints actually live -- the paths here are
# a guess based on knn_its_mycoai.py's own docstring example
# (/scratch/$USER/mycoai_models/MycoAI-{BERT,CNN}.pt), not independently
# confirmed on this cluster.
#
# Submit: sbatch slurm/final_scripts/its5m_mycoai_temp_sweep.sh
# ============================================================================
#SBATCH --job-name=its5m_mycoai_temp_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"
export WANDB_MODE=disabled
# mycoai's __init__.py calls wandb.login('allow') at import time regardless
# of WANDB_MODE, which starts a local service subprocess that writes a port
# file under $TMPDIR -- this cluster's node-local /tmp intermittently fails
# that write (confirmed repeatedly this session). Point TMPDIR at scratch
# instead, which doesn't have that flakiness.
export TMPDIR="/scratch/$USER/tmp_wandb"
mkdir -p "$TMPDIR"

mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

MYCOAI_BERT_CKPT="/scratch/$USER/mycoai_models/MycoAI-BERT.pt"
MYCOAI_CNN_CKPT="/scratch/$USER/mycoai_models/MycoAI-CNN.pt"

CKPTS=("${MYCOAI_BERT_CKPT}" "${MYCOAI_CNN_CKPT}")
TAGS=("mycoai_bert" "mycoai_cnn")

CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
echo "Checkpoint: ${CKPT} | tag: ${TAG}"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found: ${CKPT}" && exit 1

python barcodebert/knn_its_mycoai.py \
    --checkpoint               "${CKPT}" \
    --data-dir                  "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors                1 3 5 7 10 15 20 25 50 \
    --metric                     cosine \
    --knn-weights                softmax \
    --temperature-sweep          ${TEMPS} \
    --run-name                    "knn_its_${TAG}_softmax_sweep" \
    --results-file                 results_final/KNN_ITS_external_temp_sweep_RESULTS.txt
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
