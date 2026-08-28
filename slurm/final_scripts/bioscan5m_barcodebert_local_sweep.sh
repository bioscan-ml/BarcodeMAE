#!/bin/bash
# ============================================================================
# BIOSCAN-5M-pretrained BarcodeBERT baseline (the "BIOSCAN-5M" row in
# tab:bioscan_external, uniform KNN Acc.=58.31) -- a LOCAL checkpoint trained
# via this repo's own pretraining.py, not an external HF model. Runs the
# softmax-KNN temperature sweep (to fill in that row's new Softmax KNN
# column) and the ZSC eval (that row currently has no BIN AMI/HM at all).
#
# Submit: sbatch slurm/final_scripts/bioscan5m_barcodebert_local_sweep.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_barcodebert_local_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

mkdir -p results_final
mkdir -p "final_logs/${SLURM_JOB_ID}"

DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
CKPT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/external/best_pretraining.pt"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

echo "=== SOFTMAX TEMPERATURE SWEEP ==="
python barcodebert/knn_probing.py \
    --pretrained-checkpoint  "${CKPT}" \
    --dataset                 "${DATASET}" \
    --data-dir                "${DATA_DIR}" \
    --taxon                   genus \
    --representation_type     tokens \
    --n-neighbors              1 3 5 7 10 15 20 25 50 \
    --metric                   cosine \
    --knn-weights              softmax \
    --temperature-sweep        ${TEMPS} \
    --run-name                  "knn_external_barcodebert_bioscan5m_softmax_sweep" \
    --results-file               results_final/KNN_external_temp_sweep_RESULTS.txt
EC1=$?; [ ${EC1} -ne 0 ] && echo "ERROR: temperature sweep failed"

echo "=== ZSC EVALUATION ==="
python barcodebert/zsc_evaluation_v2.py \
    --pretrained-checkpoint  "${CKPT}" \
    --dataset                 "${DATASET}" \
    --data-dir                "${DATA_DIR}" \
    --taxon                   genus \
    --representation_type     tokens \
    --n-neighbors              15 \
    --metric                   cosine \
    --run-name                  "zsc_external_barcodebert_bioscan5m" \
    --results-file               results_final/ZSC_external_RESULTS.txt
EC2=$?; [ ${EC2} -ne 0 ] && echo "ERROR: ZSC eval failed"

OVERALL_EXIT=0
[ ${EC1} -ne 0 ] && OVERALL_EXIT=${EC1}
[ ${EC2} -ne 0 ] && OVERALL_EXIT=${EC2}
echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}