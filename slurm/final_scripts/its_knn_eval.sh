#!/bin/bash
# ============================================================================
# ITS-5M KNN evaluation — all 10 pretrained encoders, no finetuning
#
#   Task | Arch        | CLS | Aux    | Repr types
#   -----|-------------|-----|--------|---------------------------
#     0  | maelm       | no  | none   | tokens
#     1  | maelm       | yes | none   | tokens, cls, tokens_with_cls
#     2  | maelm       | yes | binary | tokens, cls, tokens_with_cls
#     3  | maelm       | yes | triplet| tokens, cls, tokens_with_cls
#     4  | maelm       | yes | ce     | tokens, cls, tokens_with_cls
#     5  | transformer | no  | none   | tokens
#     6  | transformer | yes | none   | tokens, cls, tokens_with_cls
#     7  | transformer | yes | binary | tokens, cls, tokens_with_cls
#     8  | transformer | yes | triplet| tokens, cls, tokens_with_cls
#     9  | transformer | yes | ce     | tokens, cls, tokens_with_cls
#
# Checkpoints from job 43413850:
#   BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ITS-5M/
# Results: results_final/KNN_ITS_RESULTS.txt
# ============================================================================
#SBATCH --job-name=its_knn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%4
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Node $SLURMD_NODENAME | $(date)"

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

# ── Sweep grid ────────────────────────────────────────────────────────────────
ARCHS=(    "maelm"  "maelm"  "maelm"  "maelm"  "maelm"  "transformer" "transformer" "transformer" "transformer" "transformer")
HAS_CLS=(  "no"     "yes"    "yes"    "yes"    "yes"    "no"          "yes"         "yes"         "yes"         "yes"        )
AUX_TASKS=("none"   "none"   "binary" "triplet" "ce"    "none"        "none"        "binary"      "triplet"     "ce"         )

ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
HAS_CLS_VAL="${HAS_CLS[$SLURM_ARRAY_TASK_ID]}"
AUX_TASK="${AUX_TASKS[$SLURM_ARRAY_TASK_ID]}"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
K_MER=6

[ "$HAS_CLS_VAL" = "no" ] && CLS_LABEL="nocls" || CLS_LABEL="cls_${AUX_TASK}"
if [ "$ARCH" = "maelm" ]; then
    RUN_NAME="final_its_k${K_MER}_6L6H_6DL6DH_maelm_${CLS_LABEL}"
else
    RUN_NAME="final_its_k${K_MER}_6L6H_transformer_${CLS_LABEL}"
fi

CKPT_BASE="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/${DATASET}/${RUN_NAME}"
[ "$ARCH" = "maelm" ] && CKPT="${CKPT_BASE}/checkpoint_encoder.pt" || CKPT="${CKPT_BASE}/checkpoint.pt"

echo "Task: ${SLURM_ARRAY_TASK_ID} | Arch: ${ARCH} | CLS: ${HAS_CLS_VAL} | Aux: ${AUX_TASK}"
echo "Run:  ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"

[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

[ "$HAS_CLS_VAL" = "no" ] && REP_TYPES=("tokens") || REP_TYPES=("tokens" "cls" "tokens_with_cls")

OVERALL_EXIT=0
for REPR in "${REP_TYPES[@]}"; do
    echo "--- KNN repr=${REPR} ---"
    python barcodebert/knn_its.py \
        --pretrained-checkpoint "${CKPT}"                          \
        --data-dir              "${DATA_DIR}"                      \
        --run-name              "knn_its_${RUN_NAME}_${REPR}"      \
        --n-neighbors           1 3 5 7                            \
        --metric                cosine                             \
        --representation-type   "${REPR}"                          \
        --results-file          results_final/KNN_ITS_RESULTS.txt  \
        --log-wandb                                                \
        --wandb-project         barcodemae_cls

    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for repr=${REPR}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}
