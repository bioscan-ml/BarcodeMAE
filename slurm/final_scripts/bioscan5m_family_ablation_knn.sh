#!/bin/bash
# ============================================================================
# BIOSCAN-5M family-level auxiliary-objective ablation — KNN eval ONLY
# (k=1), against checkpoints already trained by ablation_taxonomy_level.sh.
#
# ablation_taxonomy_level.sh trained all 6 (arch x objective) configs at
# family level (and order, which we're no longer reporting), but its inline
# KNN/ZSC eval step didn't complete for all of them (results_bioscan5m/
# KNN_RESULTS_final_abl_taxalevel.txt is missing CE and some other rows).
# This script re-evaluates KNN only (no retraining, no ZSC) at k=1 for all
# 6 configs x 3 representations, since the checkpoints already exist.
#
# 6 array tasks (0-5): (maelm|transformer) x (binary|triplet|ce)
# Each task loops over tokens/cls/tokens_with_cls internally.
#
# Results: results_final/KNN_RESULTS_final_abl_family_k1.txt
#
# Submit:  sbatch slurm/final_scripts/bioscan5m_family_ablation_knn.sh
# ============================================================================
#SBATCH --job-name=bioscan_family_knn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"

DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Grid (6 tasks): matches ablation_taxonomy_level.sh's family-level configs ─
ARCHS=("maelm" "maelm" "maelm" "transformer" "transformer" "transformer")
AUX_TASKS=("binary" "triplet" "ce" "binary" "triplet" "ce")

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"

if [ "${ARCH}" = "maelm" ]; then
    RUN_NAME="abl_taxafamily_k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_${ARCH}_cls_${AUX_TASK}"
    CFILE="checkpoint_encoder.pt"
else
    RUN_NAME="abl_taxafamily_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_cls_${AUX_TASK}"
    CFILE="checkpoint.pt"
fi

CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/taxonomy_level/${DATASET}/${RUN_NAME}/${CFILE}"
echo "Arch: ${ARCH} | Aux: ${AUX_TASK} | Ckpt: ${CKPT}"
[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

OVERALL_EXIT=0
for REP_TYPE in "tokens" "cls" "tokens_with_cls"; do
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --representation_type "${REP_TYPE}" --taxon genus --n-neighbors 1 \
        --run-name "knn_${RUN_NAME}_${REP_TYPE}" \
        --results-file results_final/KNN_RESULTS_final_abl_family_k1.txt \
        --wandb-project "${WANDB_PROJECT}" --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for ${REP_TYPE}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}