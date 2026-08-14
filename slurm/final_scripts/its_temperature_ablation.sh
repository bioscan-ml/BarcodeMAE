#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) softmax-voting TEMPERATURE ablation, for the best ITS-5M
# configuration (encoder-decoder, +CLS+Binary, CLS representation). Sweeps
# --temperature over a range of values (the main text uses T=0.07,
# DINOv2's default) to see how sensitive softmax-voting stability/accuracy
# is to this choice. T=0.07 itself is NOT included -- that result already
# exists from the main sweep, no need to recompute it.
#
# REQUIRES:
#   - its_export_tasks.sh already run (produces data/ITS-5M/tasks/test{1,2}_tasks.csv)
#   - The best ITS-5M checkpoint present at the path below (checkpoint_encoder.pt
#     for the encoder-decoder, +CLS+Binary config) -- see this script's own
#     CKPT path for the exact expected location on this cluster.
#
# 7 array tasks (0-6), one per temperature value. Each task runs leakage-free
# genus-level KNN (softmax voting only -- uniform voting doesn't use
# temperature at all, so it's identical across every task and already
# reported in the main results) at k=1,3,5,7,10,15,20,25,50, cosine metric,
# on both Yeast and Filamentous in one pass.
#
# Results: results_final/KNN_ITS_temperature_ablation_RESULTS.txt (auto-routed
# to KNN_softmax_ITS_temperature_ablation_RESULTS.txt by knn_results_path()).
#
# Submit: sbatch slurm/final_scripts/its_temperature_ablation.sh
# ============================================================================
#SBATCH --job-name=its_temp_ablation
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
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
export WANDB_DIR="/project/6045013/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATASET="ITS-5M"
DATA_DIR="/project/6045013/m4safari/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

# Best ITS-5M config: encoder-decoder (maelm), +CLS+Binary, CLS representation.
# Same naming convention as its_knn_clean_softmaxeval.sh.
CKPT="/project/6045013/m4safari/BarcodeMAE/main_checkpoints_final/${DATASET}/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT} — move it from the ITS-5M cluster first" && exit 1

# ── Grid (8 tasks): temperature values ──────────────────────────────────────
TEMPERATURES=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
TEMPERATURE="${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}"

echo "Temperature: ${TEMPERATURE}"

python barcodebert/knn_its_clean.py \
    --pretrained-checkpoint "${CKPT}" \
    --data-dir              "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
    --representation-type   cls \
    --tasks                 genus_level \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            softmax --temperature ${TEMPERATURE} \
    --run-name                "temp_ablation_T${TEMPERATURE}" \
    --results-file             results_final/KNN_ITS_temperature_ablation_RESULTS.txt \
    --log-wandb --wandb-project "${WANDB_PROJECT}"
EC=$?

[ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for T=${TEMPERATURE}"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}