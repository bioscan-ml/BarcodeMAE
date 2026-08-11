#!/bin/bash
# ============================================================================
# BIOSCAN-5M external baseline comparison — zero-shot evaluation of
# off-the-shelf HuggingFace DNA foundation model checkpoints (no
# fine-tuning), via external_models.py's adapter (see that file's docstring
# for why no changes were needed to knn_probing.py/zsc_evaluation_v2.py's
# actual embedding-extraction logic, only their model-loading front end).
#
# NOT included here: BarcodeMamba. Its published checkpoint is a plain
# GitHub repo with its own training code (bioscan-ml/BarcodeMamba), not a
# HuggingFace AutoModel-compatible checkpoint, so it doesn't fit the generic
# adapter in external_models.py -- it needs a separate, custom loader that
# hasn't been written yet.
#
# Representation: always "tokens" (attention-mask-aware mean pooling over
# the external model's own hidden states) -- external checkpoints don't share
# our CLS-token/Jumbo-token setup, so there's no meaningful cls/tokens_with_cls
# variant to sweep per model, unlike our own configs.
#
# 9 array tasks (0-8), one per baseline model. Each task runs KNN (uniform +
# softmax voting, k=1,3,5,7,10,15,20,25,50, cosine metric, T=0.07) and ZSC
# (k=15) on genus-level BIOSCAN-5M.
#
# Results: results_final/KNN_external_RESULTS.txt (uniform, auto-routed to
# KNN_softmax_external_RESULTS.txt for softmax) and
# results_final/ZSC_external_RESULTS.txt.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_external_baselines.sh
# ============================================================================
#SBATCH --job-name=bioscan_external
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
TEMPERATURE=0.07
MAX_LEN=660

# ── Grid (7 tasks): HF repo id | HF auto-class | short tag ────────────────────
# GROVER and Omni-DNA dropped: encoder-only GENA-LM is the architecturally
# closer/more important comparison point (matches BarcodeMAE+'s encoder-based
# MAE setup), and Omni-DNA (causal, arXiv-only preprint) was judged lowest
# priority of the three.
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
    "auto"       # DNABERT-2
    "auto"       # DNABERT-S
    "masked-lm"  # Nucleotide Transformer
    "auto"       # BarcodeBERT
    "causal-lm"  # HyenaDNA-tiny
    "masked-lm"  # Caduceus-PS-1k
    "auto"       # GENA-LM (ModernGENA)
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

# ── KNN (uniform + softmax voting, k = 1,3,5,7,10,15,20,25,50) ────────────────
echo "=== KNN EVALUATION ==="
for WEIGHTS in "uniform" "softmax"; do
    WEIGHT_ARGS=(--knn-weights "${WEIGHTS}")
    [ "${WEIGHTS}" = "softmax" ] && WEIGHT_ARGS+=(--temperature ${TEMPERATURE})
    python barcodebert/knn_probing.py \
        --external-model-id     "${MODEL_ID}" \
        --external-model-cls    "${MODEL_CLS_ARG}" \
        --external-max-length   ${MAX_LEN} \
        --dataset                "${DATASET}" \
        --data-dir               "${DATA_DIR}" \
        --taxon                  genus \
        --n-neighbors             1 3 5 7 10 15 20 25 50 \
        --metric                  cosine \
        "${WEIGHT_ARGS[@]}" \
        --run-name                "knn_external_${TAG}_${WEIGHTS}" \
        --results-file             results_final/KNN_external_RESULTS.txt \
        --wandb-project           "${WANDB_PROJECT}" \
        --log-wandb
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: KNN failed for ${TAG}/${WEIGHTS}" && OVERALL_EXIT=${EC}
done

# ── ZSC ───────────────────────────────────────────────────────────────────────
echo "=== ZSC EVALUATION ==="
python barcodebert/zsc_evaluation_v2.py \
    --external-model-id     "${MODEL_ID}" \
    --external-model-cls    "${MODEL_CLS_ARG}" \
    --external-max-length   ${MAX_LEN} \
    --dataset                "${DATASET}" \
    --data-dir                "${DATA_DIR}" \
    --taxon                   genus \
    --n-neighbors              15 \
    --metric                   cosine \
    --run-name                 "zsc_external_${TAG}" \
    --results-file             results_final/ZSC_external_RESULTS.txt \
    --wandb-project            "${WANDB_PROJECT}" \
    --log-wandb
EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: ZSC failed for ${TAG}" && OVERALL_EXIT=${EC}

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}