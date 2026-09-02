#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) external baseline comparison, WITH leakage included --
# counterpart to its5m_external_baselines.sh. Same models, same KNN pipeline
# (knn_its_clean.py, unchanged), only --tasks-dir differs: points at
# tasks_with_leakage/ (exact/substring duplicates included in the query
# pools) instead of tasks/ (leakage-free). See its_export_tasks_with_leakage.sh
# and analyze_its_overlap.py's --include-leaked flag.
#
# REQUIRES its_export_tasks_with_leakage.sh to have been run first (produces
# data/ITS-5M/tasks_with_leakage/test{1,2}_tasks.csv).
#
# Results: results_final/KNN_ITS_external_with_leakage_RESULTS.txt (uniform,
# auto-routed to KNN_softmax_ITS_external_with_leakage_RESULTS.txt for softmax).
#
# Submit: sbatch slurm/final_scripts/its5m_external_baselines_with_leakage.sh
# ============================================================================
#SBATCH --job-name=its5m_external_with_leakage
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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

# See its5m_external_baselines.sh for the shared-venv pip-install warning --
# same rule applies here (don't pip install inside an array job).

export WANDB_MODE=offline
export WANDB_DIR="/project/6045013/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks_with_leakage"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks_with_leakage.sh first" && exit 1
TEMPERATURE=0.07
MAX_LEN=660

# ── Grid (7 tasks): same models as its5m_external_baselines.sh ────────────
MODEL_IDS=(
    "zhihan1996/DNABERT-2-117M"
    "zhihan1996/DNABERT-S"
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    "bioscan-ml/BarcodeBERT"
    "LongSafari/hyenadna-tiny-1k-seqlen"
    "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
    "AIRI-Institute/moderngena-base"
)
MODEL_CLS=(
    "auto" "auto" "masked-lm" "auto" "causal-lm" "masked-lm" "auto"
)
MODEL_TAGS=(
    "dnabert2" "dnaberts" "nucleotide_transformer" "barcodebert"
    "hyenadna_tiny" "caduceus_ps1k" "gena_lm"
)

MODEL_ID="${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}"
MODEL_CLS_ARG="${MODEL_CLS[$SLURM_ARRAY_TASK_ID]}"
TAG="${MODEL_TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "Model: ${MODEL_ID} | class: ${MODEL_CLS_ARG} | tag: ${TAG}"

OVERALL_EXIT=0
for WEIGHTS in "uniform" "softmax"; do
    WEIGHT_ARGS=(--knn-weights "${WEIGHTS}")
    [ "${WEIGHTS}" = "softmax" ] && WEIGHT_ARGS+=(--temperature ${TEMPERATURE})
    python barcodebert/knn_its_clean.py \
        --external-model-id     "${MODEL_ID}" \
        --external-model-cls    "${MODEL_CLS_ARG}" \
        --external-max-length   ${MAX_LEN} \
        --data-dir                "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --representation-type     tokens \
        --n-neighbors              1 3 5 7 10 15 20 25 50 \
        --metric                   cosine \
        --embed-batch-size         32 \
        "${WEIGHT_ARGS[@]}" \
        --run-name                 "knn_external_${TAG}_${WEIGHTS}_with_leakage" \
        --results-file             results_final/KNN_ITS_external_with_leakage_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for ${TAG}/${WEIGHTS}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}