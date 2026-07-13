#!/bin/bash
# ============================================================================
# Random-Initialized Encoder Baseline — KNN, no pretraining
#
# Sanity-check baseline: run KNN eval with an UNTRAINED encoder to confirm
# that pretrained checkpoints actually learn something beyond random
# projections. Uses random_knn.py (BIOSCAN-5M) / random_knn_its.py (fungi
# ITS-5M), representation_type=tokens, no CLS token — matching the "nocls"
# baseline config used in the main experiments (bioscan5m_final.sh /
# fungi_its_final.sh, task 0/5).
#
# 2 tasks (0-1): maelm, transformer — same arch grid as the other ablation
# scripts. DATASET is an env var like the rest of slurm/final_scripts/*.sh.
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M random_baseline_knn.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     random_baseline_knn.sh
#
# Results: results_final/RANDOM_KNN_RESULTS.txt (BIOSCAN-5M)
#          results_final/RANDOM_KNN_ITS_RESULTS.txt (ITS-5M)
# ============================================================================
#SBATCH --job-name=random_baseline_knn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Dataset: ${DATASET:-BIOSCAN-5M} | $(date)"

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

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

# ── Grid ──────────────────────────────────────────────────────────────────────
ARCHS=("maelm" "transformer")
ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

# ── Dataset (overridable via --export=DATASET=...) ────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
fi

K_MER=6; STRIDE=6; N_LAYERS=6; N_HEADS=6; ENCODER_DIM=768

RUN_NAME="random_k${K_MER}_${N_LAYERS}L${N_HEADS}H_${ARCH}_nocls"

echo "Dataset: ${DATASET} | Arch: ${ARCH} | Run: ${RUN_NAME}"

if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    python barcodebert/random_knn.py \
        --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE} \
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS} --encoder-embed-dim ${ENCODER_DIM} \
        --taxon genus --n-neighbors 1 3 5 7 --representation-type tokens \
        --run-name "${RUN_NAME}" \
        --results-file results_final/RANDOM_KNN_RESULTS.txt
    EC=$?
else
    python barcodebert/random_knn_its.py \
        --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${STRIDE} \
        --n-layers ${N_LAYERS} --n-heads ${N_HEADS} --encoder-embed-dim ${ENCODER_DIM} \
        --n-neighbors 1 3 5 7 --representation-type tokens \
        --run-name "${RUN_NAME}" \
        --results-file results_final/RANDOM_KNN_ITS_RESULTS.txt
    EC=$?
fi

[ ${EC} -ne 0 ] && echo "ERROR: random baseline KNN failed"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}