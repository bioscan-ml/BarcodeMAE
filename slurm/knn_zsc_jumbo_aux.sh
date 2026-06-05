#!/bin/bash
#SBATCH --job-name=knn_zsc_jumbo_aux
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

# KNN + ZSC evaluation for BIOSCAN-5M jumbo aux-loss checkpoints.
# Representation types evaluated: jumbo_avg, jumbo
#
#  task 0: MAELM  + triplet k16 mg0.0  — jumbo_avg
#  task 1: MAELM  + triplet k16 mg0.0  — jumbo
#  task 2: MAELM  + supcon  k8  τ=0.15 — jumbo_avg
#  task 3: MAELM  + supcon  k8  τ=0.15 — jumbo
#  task 4: MAELM  + ce      k8         — jumbo_avg  (add when ready)
#  task 5: MAELM  + ce      k8         — jumbo      (add when ready)

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/model_checkpoints/BIOSCAN-5M/aux_jumbo"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
DATASET="BIOSCAN-5M"

# ── Checkpoint × representation type grid ─────────────────────────────────────
CKPT_NAMES=(
    "maelm_jumbo6x2_auxtriplet_k16_mg0p0"
    "maelm_jumbo6x2_auxtriplet_k16_mg0p0"
    "maelm_jumbo6x2_auxsupcon_k8_t015"
    "maelm_jumbo6x2_auxsupcon_k8_t015"
    "maelm_jumbo6x2_auxce_k8"
    "maelm_jumbo6x2_auxce_k8"
)

CHECKPOINTS=(
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxtriplet_genus_km16x4_mg0p0/checkpoint_encoder.pt"
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxtriplet_genus_km16x4_mg0p0/checkpoint_encoder.pt"
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxsupcon_genus_km8x4_mg0p3/checkpoint_encoder.pt"
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxsupcon_genus_km8x4_mg0p3/checkpoint_encoder.pt"
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxce_genus_km8x4_mg0p3/checkpoint_encoder.pt"
    "${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_maelm_jumbo6x2_auxce_genus_km8x4_mg0p3/checkpoint_encoder.pt"
)

REP_TYPES=("jumbo_avg" "jumbo" "jumbo_avg" "jumbo" "jumbo_avg" "jumbo")

CKPT_NAME="${CKPT_NAMES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
REP_TYPE="${REP_TYPES[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Run:        ${CKPT_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Rep type:   ${REP_TYPE}"
echo "=========================================="

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}

OVERALL_EXIT=0

# ── KNN ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- KNN (representation=${REP_TYPE}) ---"

python barcodebert/knn_probing.py \
    --pretrained-checkpoint "${CHECKPOINT}" \
    --dataset               "${DATASET}"    \
    --data-dir              "${DATA_DIR}"   \
    --representation_type   "${REP_TYPE}"  \
    --taxon genus \
    --n-neighbors 1 \
    --run-name "knn_${CKPT_NAME}_${REP_TYPE}" \
    --log-wandb

EXIT_CODE=$?
[ ${EXIT_CODE} -ne 0 ] && echo "ERROR: KNN failed (exit ${EXIT_CODE})" && OVERALL_EXIT=${EXIT_CODE}

# ── ZSC ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- ZSC (representation=${REP_TYPE}) ---"

python barcodebert/zsc_evaluation_v2.py \
    --pretrained-checkpoint "${CHECKPOINT}" \
    --dataset               "${DATASET}"    \
    --data-dir              "${DATA_DIR}"   \
    --representation_type   "${REP_TYPE}"  \
    --taxon genus \
    --n-neighbors 15 \
    --metric cosine \
    --run-name "zsc_${CKPT_NAME}_${REP_TYPE}" \
    --log-wandb

EXIT_CODE=$?
[ ${EXIT_CODE} -ne 0 ] && echo "ERROR: ZSC failed (exit ${EXIT_CODE})" && OVERALL_EXIT=${EXIT_CODE}

echo "=========================================="
echo "Job finished at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="

exit ${OVERALL_EXIT}