#!/bin/bash
# ============================================================================
# BIOSCAN-5M softmax-voting TEMPERATURE ablation -- T=0.07 ONLY.
#
# Counterpart to bioscan5m_temperature_ablation.sh (which deliberately
# excludes T=0.07, assuming it "already exists from the main sweep"). That
# assumption turned out to be wrong: the only existing T=0.07 result was
# generated against the stale alpha=0.1 checkpoint, so Section C's BIOSCAN-5M
# table still has one stale row (75.24/74.03/1.14/3.92) even after every
# other temperature was re-run against the correct alpha=1.0 checkpoint.
# This single-task script fills that one remaining gap.
#
# REQUIRES: the alpha=1.0 CE checkpoint (same one
# bioscan5m_temperature_ablation.sh uses).
#
# Results: results_final/KNN_bioscan5m_temperature_ablation_RESULTS.txt --
# same file as bioscan5m_temperature_ablation.sh, so this row lands
# alongside the other 7 temperatures already in it.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_temperature_ablation_t007.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_temp_t007
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=final_logs/%A/%A.out
#SBATCH --error=final_logs/%A/%A.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/job_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"

# Same alpha=1.0 CE checkpoint as bioscan5m_temperature_ablation.sh.
CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight/${DATASET}/ablw_bioscan5m_k6_6L6H_6DL6DH_maelm_cls_ce_w1.0/checkpoint_encoder.pt"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

TEMPERATURE=0.07
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