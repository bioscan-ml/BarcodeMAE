#!/bin/bash
#SBATCH --job-name=zsc_aux
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-17%6
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

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

VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"

echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/model_checkpoints/BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
DATASET="BIOSCAN-5M"

# ── Checkpoint table ──────────────────────────────────────────────────────────
# Format: RUN_NAME|CHECKPOINT_PATH|REP_TYPE
#
# aux_3/ — original aux loss runs (tasks 0-5)
#   0: MAELM  triplet  k8m4 mg0.3
#   1: MAELM  supcon   k8m4 τ=0.07
#   2: MAELM  ce       k8m4
#   3: Transformer triplet  k8m4 mg0.3
#   4: Transformer supcon   k8m4 τ=0.07
#   5: Transformer ce       k8m4
#
# aux_4/ — rerun supcon with τ=0.15 + warmup (tasks 6-7)
#   6: MAELM  supcon   k8m4 τ=0.15
#   7: Transformer supcon   k8m4 τ=0.15
#
# aux_sweep_triplet/ — triplet hyperparameter sweep (tasks 8-17)
#   8-12:  MAELM       k/margin combos
#   13-17: Transformer k/margin combos

MAELM_AUX3="${CKPT_ROOT}/aux_3"
TRANS_AUX3="${CKPT_ROOT}/aux_3"
MAELM_AUX4="${CKPT_ROOT}/aux_4"
TRANS_AUX4="${CKPT_ROOT}/aux_4"
SWEEP="${CKPT_ROOT}/aux_sweep_triplet"

RUN_NAMES=(
    # aux_3 MAELM
    "maelm_triplet_k8_mg0p3_aux3"
    "maelm_supcon_k8_t007_aux3"
    "maelm_ce_k8_aux3"
    # aux_3 transformer
    "transformer_triplet_k8_mg0p3_aux3"
    "transformer_supcon_k8_t007_aux3"
    "transformer_ce_k8_aux3"
    # aux_4 supcon reruns
    "maelm_supcon_k8_t015_aux4"
    "transformer_supcon_k8_t015_aux4"
    # aux_sweep_triplet MAELM (tasks 0-4 of triplet_sweep.sh)
    "maelm_triplet_k8_mg0p5_sweep"
    "maelm_triplet_k16_mg0p3_sweep"
    "maelm_triplet_k16_mg0p5_sweep"
    "maelm_triplet_k8_mg0p0_sweep"
    "maelm_triplet_k16_mg0p0_sweep"
    # aux_sweep_triplet transformer (tasks 0-4 of triplet_sweep_transformer.sh)
    "transformer_triplet_k8_mg0p5_sweep"
    "transformer_triplet_k16_mg0p3_sweep"
    "transformer_triplet_k16_mg0p5_sweep"
    "transformer_triplet_k8_mg0p0_sweep"
    "transformer_triplet_k16_mg0p0_sweep"
)

CHECKPOINTS=(
    "${MAELM_AUX3}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km8x4/checkpoint_encoder.pt"
    "${MAELM_AUX3}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxsupcon_genus_km8x4/checkpoint_encoder.pt"
    "${MAELM_AUX3}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxce_genus_km8x4/checkpoint_encoder.pt"
    "${TRANS_AUX3}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km8x4/checkpoint.pt"
    "${TRANS_AUX3}/run_k6_6L_6H_transformer_cls_auxsupcon_genus_km8x4/checkpoint.pt"
    "${TRANS_AUX3}/run_k6_6L_6H_transformer_cls_auxce_genus_km8x4/checkpoint.pt"
    "${MAELM_AUX4}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxsupcon_genus_km8x4/checkpoint_encoder.pt"
    "${TRANS_AUX4}/run_k6_6L_6H_transformer_cls_auxsupcon_genus_km8x4/checkpoint.pt"
    "${SWEEP}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km8x4_mg0p5/checkpoint_encoder.pt"
    "${SWEEP}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km16x4_mg0p3/checkpoint_encoder.pt"
    "${SWEEP}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km16x4_mg0p5/checkpoint_encoder.pt"
    "${SWEEP}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km8x4_mg0p0/checkpoint_encoder.pt"
    "${SWEEP}/run_k6_6L_6H_6DL_6DH_maelm_cls_auxtriplet_genus_km16x4_mg0p0/checkpoint_encoder.pt"
    "${SWEEP}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km8x4_mg0p5/checkpoint.pt"
    "${SWEEP}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km16x4_mg0p3/checkpoint.pt"
    "${SWEEP}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km16x4_mg0p5/checkpoint.pt"
    "${SWEEP}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km8x4_mg0p0/checkpoint.pt"
    "${SWEEP}/run_k6_6L_6H_transformer_cls_auxtriplet_genus_km16x4_mg0p0/checkpoint.pt"
)

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Run:        ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

mkdir -p final_logs/${SLURM_ARRAY_JOB_ID}

OVERALL_EXIT=0

REP_TYPES=("cls" "tokens_with_cls")

for REP_TYPE in "${REP_TYPES[@]}"; do
    echo ""
    echo "--- Representation type: ${REP_TYPE} ---"

    python barcodebert/zsc_evaluation_v2.py \
        --pretrained-checkpoint "${CHECKPOINT}" \
        --dataset "${DATASET}" \
        --data-dir "${DATA_DIR}" \
        --representation_type "${REP_TYPE}" \
        --taxon genus \
        --n-neighbors 15 \
        --metric cosine \
        --run-name "zsc_${RUN_NAME}_${REP_TYPE}" \
        --log-wandb

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: zsc_evaluation_v2 failed for ${RUN_NAME} / ${REP_TYPE} (exit ${EXIT_CODE})"
        OVERALL_EXIT=${EXIT_CODE}
    fi
done

echo "=========================================="
echo "Job finished at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="

exit ${OVERALL_EXIT}