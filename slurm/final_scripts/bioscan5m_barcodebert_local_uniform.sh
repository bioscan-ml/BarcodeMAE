#!/bin/bash
# ============================================================================
# BIOSCAN-5M-pretrained BarcodeBERT baseline -- uniform-KNN ONLY. Softmax
# sweep and ZSC for this checkpoint were already run separately
# (bioscan5m_barcodebert_local_sweep.sh); this just fills in the real
# 1-NN Acc. value for tab:bioscan_external's BIOSCAN-5M/BarcodeBERT row.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_barcodebert_local_uniform.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_barcodebert_local_uniform
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=final_logs/%j/%j.out
#SBATCH --error=final_logs/%j/%j.err

echo "Job $SLURM_JOB_ID | $(date)"

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
mkdir -p "final_logs/${SLURM_JOB_ID}"

DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/external/best_pretraining.pt"

echo "=== UNIFORM KNN EVALUATION ==="
python barcodebert/knn_probing.py \
    --pretrained-checkpoint  "${CKPT}" \
    --dataset                 "${DATASET}" \
    --data-dir                "${DATA_DIR}" \
    --taxon                   genus \
    --representation_type     tokens \
    --n-neighbors              1 3 5 7 10 15 20 25 50 \
    --metric                   cosine \
    --knn-weights              uniform \
    --run-name                  "knn_external_barcodebert_bioscan5m_uniform" \
    --results-file               results_final/KNN_external_temp_sweep_RESULTS.txt
EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: uniform KNN eval failed"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}