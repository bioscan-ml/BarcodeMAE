#!/bin/bash
# ============================================================================
# ITS-5M: softmax-KNN temperature sweep for the BIOSCAN-5M-pretrained
# BarcodeMamba+ checkpoint (cross-dataset generalization check), WITH
# leakage included -- counterpart to its5m_barcodemamba_bioscan5mckpt_sweep.sh.
# Same checkpoint, same KNN pipeline, only --tasks-dir differs (points at
# tasks_with_leakage/ instead of tasks/).
#
# *** CHECKPOINT PATH: must match whatever you used in
# bioscan5m_barcodemamba_sweep.sh's CHECKPOINT_DIR.
#
# REQUIRES slurm/setup_env_barcodemamba.sh to have been run once first, AND
# its_export_tasks_with_leakage.sh to have been run first.
#
# Submit: sbatch slurm/final_scripts/its5m_barcodemamba_bioscan5mckpt_sweep_with_leakage.sh
# ============================================================================
#SBATCH --job-name=its5m_barcodemamba_bioscan5mckpt_sweep_with_leakage
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
CHECKPOINT_DIR="/scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M"
BPE_TOKENIZER="${CHECKPOINT_DIR}/bpe_tokenizer.pkl"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks_with_leakage"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks_with_leakage.sh first" && exit 1
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

echo "=== UNIFORM KNN EVALUATION (BIOSCAN-5M checkpoint on ITS-5M) ==="
python barcodebert/knn_its_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            uniform \
    --run-name                knn_its_barcodemamba_bioscan5mckpt_uniform_with_leakage \
    --results-file             results_final/KNN_ITS_external_with_leakage_temp_sweep_RESULTS.txt
EC0=$?; [ ${EC0} -ne 0 ] && echo "ERROR: uniform KNN eval failed"

echo "=== SOFTMAX TEMPERATURE SWEEP (BIOSCAN-5M checkpoint on ITS-5M) ==="
python barcodebert/knn_its_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            softmax \
    --temperature-sweep      ${TEMPS} \
    --run-name                knn_its_barcodemamba_bioscan5mckpt_softmax_sweep_with_leakage \
    --results-file             results_final/KNN_ITS_external_with_leakage_temp_sweep_RESULTS.txt
EC1=$?; [ ${EC1} -ne 0 ] && echo "ERROR: temperature sweep failed"

OVERALL_EXIT=0
[ ${EC0} -ne 0 ] && OVERALL_EXIT=${EC0}
[ ${EC1} -ne 0 ] && OVERALL_EXIT=${EC1}
echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}