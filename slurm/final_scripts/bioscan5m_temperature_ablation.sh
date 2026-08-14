#!/bin/bash
# ============================================================================
# BIOSCAN-5M softmax-voting TEMPERATURE ablation, for the best BIOSCAN-5M
# configuration (encoder-decoder, +CLS+CE, CLS representation). Sweeps
# --temperature over a range of values (the main text uses T=0.07, DINOv2's
# default) to see how sensitive softmax-voting stability/accuracy is to this
# choice. T=0.07 itself is NOT included -- that result already exists from
# the main sweep, no need to recompute it.
#
# REQUIRES: the best BIOSCAN-5M checkpoint present at the CKPT path below
# (checkpoint_encoder.pt for the encoder-decoder, +CLS+CE config).
#
# 7 array tasks (0-6), one per temperature value. Each task runs genus-level
# KNN (softmax voting only) at k=1,3,5,7,10,15,20,25,50, cosine metric.
#
# Results: results_final/KNN_bioscan5m_temperature_ablation_RESULTS.txt
# (auto-routed to KNN_softmax_bioscan5m_temperature_ablation_RESULTS.txt by
# knn_results_path()).
#
# Submit: sbatch slurm/final_scripts/bioscan5m_temperature_ablation.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_temp_ablation
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
DATASET="BIOSCAN-5M"
DATA_DIR="/project/6045013/m4safari/BarcodeMAE/data/${DATASET}"

# Best BIOSCAN-5M config: encoder-decoder (maelm), +CLS+CE, CLS representation.
# Same naming convention as bioscan5m_softmaxknn_eval.sh.
CKPT="/project/6045013/m4safari/BarcodeMAE/main_checkpoints_final/${DATASET}/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

# ── Grid (7 tasks): temperature values (0.07 excluded, already have it) ─────
TEMPERATURES=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
TEMPERATURE="${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}"

echo "Temperature: ${TEMPERATURE}"

python barcodebert/knn_probing.py \
    --pretrained-checkpoint "${CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
    --representation_type   cls --taxon genus \
    --n-neighbors            1 3 5 7 10 15 20 25 50 \
    --metric                 cosine \
    --knn-weights            softmax --temperature ${TEMPERATURE} \
    --run-name                "temp_ablation_T${TEMPERATURE}" \
    --results-file             results_final/KNN_bioscan5m_temperature_ablation_RESULTS.txt \
    --wandb-project "${WANDB_PROJECT}" --log-wandb
EC=$?

[ ${EC} -ne 0 ] && echo "ERROR: knn_probing.py failed for T=${TEMPERATURE}"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}