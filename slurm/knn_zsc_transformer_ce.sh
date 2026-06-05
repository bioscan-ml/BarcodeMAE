#!/bin/bash
#SBATCH --job-name=knn_zsc_trans_ce
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=final_logs/%j/%j.out
#SBATCH --error=final_logs/%j/%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p "final_logs/${SLURM_JOB_ID}"

CHECKPOINT="./model_checkpoints/BIOSCAN-5M/run_k6_6L_6H_transformer_cls_auxce_genus_km8x4/checkpoint.pt"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
DATASET="BIOSCAN-5M"

echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

OVERALL_EXIT=0

for REP in cls tokens_with_cls; do
    echo ""
    echo "=========================================="
    echo "--- KNN  rep=${REP} ---"
    echo "=========================================="

    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CHECKPOINT}" \
        --dataset               "${DATASET}"    \
        --data-dir              "${DATA_DIR}"   \
        --representation_type   "${REP}"        \
        --taxon genus \
        --n-neighbors 1 \
        --run-name "knn_transformer_auxce_k8_${REP}" \
        --log-wandb

    [ $? -ne 0 ] && echo "ERROR: KNN failed for ${REP}" && OVERALL_EXIT=1

    echo ""
    echo "=========================================="
    echo "--- ZSC  rep=${REP} ---"
    echo "=========================================="

    python barcodebert/zsc_evaluation_v2.py \
        --pretrained-checkpoint "${CHECKPOINT}" \
        --dataset               "${DATASET}"    \
        --data-dir              "${DATA_DIR}"   \
        --representation_type   "${REP}"        \
        --taxon genus \
        --n-neighbors 15 \
        --metric cosine \
        --run-name "zsc_transformer_auxce_k8_${REP}" \
        --log-wandb

    [ $? -ne 0 ] && echo "ERROR: ZSC failed for ${REP}" && OVERALL_EXIT=1
done

echo "=========================================="
echo "Job finished at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="

exit ${OVERALL_EXIT}