#!/bin/bash
# ============================================================================
# BIOSCAN-5M: softmax-KNN temperature sweep for BarcodeMamba/BarcodeMamba+
# (https://github.com/bioscan-ml/BarcodeMamba-dev), fills in the "BarcodeMamba
# / BIOSCAN-5M" row in tab:bioscan_external (currently all "--").
#
# *** CHECKPOINT PATH: adjust CHECKPOINT_DIR below to wherever you download
# the BIOSCAN-5M BarcodeMamba+ checkpoint folder to (the Google Drive
# "barcodemamba_plus_bi..." folder: .hydra/config.yaml, last.ckpt directly in
# the folder root, bpe_tokenizer.pkl, bpe_tokenizer_meta.json).
#
# REQUIRES slurm/setup_env_barcodemamba.sh to have been run once first.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_barcodemamba_sweep.sh
# ============================================================================
#SBATCH --job-name=bioscan5m_barcodemamba_sweep
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/BIOSCAN-5M"
TEMPS="0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0"

echo "=== UNIFORM KNN EVALUATION ==="
python barcodebert/knn_probing_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir            "${DATA_DIR}" \
    --taxon                genus \
    --n-neighbors           1 3 5 7 10 15 20 25 50 \
    --metric                cosine \
    --knn-weights           uniform \
    --run-name               knn_external_barcodemamba_bioscan5m_uniform \
    --results-file            results_final/KNN_external_temp_sweep_RESULTS.txt
EC0=$?; [ ${EC0} -ne 0 ] && echo "ERROR: uniform KNN eval failed"

echo "=== SOFTMAX TEMPERATURE SWEEP ==="
python barcodebert/knn_probing_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir            "${DATA_DIR}" \
    --taxon                genus \
    --n-neighbors           1 3 5 7 10 15 20 25 50 \
    --metric                cosine \
    --knn-weights           softmax \
    --temperature-sweep     ${TEMPS} \
    --run-name               knn_external_barcodemamba_bioscan5m_softmax_sweep \
    --results-file            results_final/KNN_external_temp_sweep_RESULTS.txt
EC1=$?; [ ${EC1} -ne 0 ] && echo "ERROR: temperature sweep failed"

echo "=== ZSC EVALUATION ==="
python barcodebert/zsc_barcodemamba.py \
    --barcodemamba-repo   "${BM_REPO}" \
    --checkpoint-dir      "${CHECKPOINT_DIR}" \
    --bpe-tokenizer-path  "${BPE_TOKENIZER}" \
    --data-dir             "${DATA_DIR}" \
    --taxon                 genus \
    --n-neighbors            15 \
    --metric                 cosine \
    --run-name                zsc_external_barcodemamba_bioscan5m \
    --results-file             results_final/ZSC_external_RESULTS.txt
EC2=$?; [ ${EC2} -ne 0 ] && echo "ERROR: ZSC eval failed"

OVERALL_EXIT=0
[ ${EC0} -ne 0 ] && OVERALL_EXIT=${EC0}
[ ${EC1} -ne 0 ] && OVERALL_EXIT=${EC1}
[ ${EC2} -ne 0 ] && OVERALL_EXIT=${EC2}
echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}