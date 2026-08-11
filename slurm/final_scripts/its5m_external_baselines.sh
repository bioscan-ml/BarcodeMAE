#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) external baseline comparison — zero-shot evaluation of
# off-the-shelf HuggingFace DNA foundation model checkpoints (no
# fine-tuning), on the leakage-free Yeast and Filamentous query pools, via
# external_models.py's adapter (see bioscan5m_external_baselines.sh and
# external_models.py's docstring for the full explanation).
#
# NOT included here: BarcodeMamba+, MycoAI-BERT, MycoAI-CNN. All three are
# plain GitHub repos with their own training/inference code, not
# HuggingFace AutoModel-compatible checkpoints, so none fit the generic
# adapter in external_models.py -- they need separate, custom loaders that
# haven't been written yet.
#
# Representation: always "tokens" (attention-mask-aware mean pooling), same
# reasoning as the BIOSCAN-5M script.
#
# REQUIRES its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv).
#
# 9 array tasks (0-8), one per baseline model. Each task runs leakage-free
# genus-level KNN (uniform + softmax voting, k=1,3,5,7,10,15,20,25,50,
# cosine metric, T=0.07) on both Yeast and Filamentous in one pass (
# knn_its_clean.py evaluates all test sets together).
#
# Results: results_final/KNN_ITS_external_RESULTS.txt (uniform, auto-routed
# to KNN_softmax_ITS_external_RESULTS.txt for softmax).
#
# Submit: sbatch slurm/final_scripts/its5m_external_baselines.sh
# ============================================================================
#SBATCH --job-name=its5m_external
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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

# ── External model dependencies ────────────────────────────────────────────
# DO NOT pip install here: every array task shares the same venv and runs
# concurrently, so a pip install inside this script races with every other
# task's pip install against the same site-packages directory -- this WILL
# corrupt the venv (half-deleted packages, missing submodules like
# functorch/, broken torch/torchtext imports for every job using this venv,
# not just this one). Install once, interactively, BEFORE submitting this
# array job:
#   source /scratch/$USER/BarcodeMAE_venv/bin/activate
#   pip install --no-deps einops triton
#   pip install --no-deps mamba-ssm causal-conv1d  # best-effort; Caduceus
#       falls back to a slow path if this isn't present -- not fatal if it fails.
# If torch/torchtext ever get corrupted by a concurrent install (symptoms:
# "ImportError: cannot import name 'inf' from 'torch'", "~orch" pip warnings,
# missing functorch/), repair with:
#   pip install --force-reinstall torch==2.1.1
#   pip install --force-reinstall torchtext==0.16.2

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1
TEMPERATURE=0.07
MAX_LEN=660

# ── Grid (7 tasks): same models as bioscan5m_external_baselines.sh ────────────
# GROVER and Omni-DNA dropped, same rationale as the BIOSCAN-5M script.
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
    "auto" "auto" "auto" "auto" "causal-lm" "masked-lm" "auto"
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
        "${WEIGHT_ARGS[@]}" \
        --run-name                 "knn_external_${TAG}_${WEIGHTS}" \
        --results-file             results_final/KNN_ITS_external_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for ${TAG}/${WEIGHTS}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}