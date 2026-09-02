#!/bin/bash
# ============================================================================
# ITS-5M: softmax-KNN temperature sweep ONLY, for BarcodeMamba+ large
# (layer4-dim768) -- resubmit for its5m_barcodemamba_sweep.sh array task 1,
# whose softmax sweep didn't finish within the original walltime while its
# uniform-KNN step already completed (61.03 Yeast / 55.49 Filamentous,
# currently in tab:its_external as the BarcodeMamba+ (large) row's 1-NN
# values). Skips the uniform step entirely to avoid rerunning completed work.
#
# CHECKPOINT PATH: same as its5m_barcodemamba_sweep.sh's layer4-dim768 entry.
#
# REQUIRES slurm/setup_env_barcodemamba.sh to have been run once first.
#
# Submit: sbatch slurm/final_scripts/its5m_barcodemamba_large_softmax_only.sh
# ============================================================================
#SBATCH --job-name=its5m_barcodemamba_large_softmax_only
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=final_logs/%j/%j.out
#SBATCH --error=final_logs/%j/%j.err

echo "Job $SLURM_JOB_ID | $(date)"

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
mkdir -p "final_logs/${SLURM_JOB_ID}"

BM_REPO="/scratch/$USER/BarcodeMamba-dev"
CHECKPOINT_BASE="/scratch/$USER/barcodemamba_checkpoints/models_release"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/BarcodeMamba-plus-layer4-dim768"
BPE_TOKENIZER="${CHECKPOINT_BASE}/bpe_tokenizer.pkl"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

echo "=== SOFTMAX TEMPERATURE SWEEP (BarcodeMamba+ large, layer4-dim768) ==="
python barcodebert/knn_its_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            softmax \
    --temperature-sweep      ${TEMPS} \
    --run-name                "knn_its_barcodemamba_layer4dim768_softmax_sweep" \
    --results-file             results_final/KNN_ITS_external_temp_sweep_RESULTS.txt
EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: softmax sweep failed"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}