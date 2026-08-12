#!/bin/bash
# ============================================================================
# BIOSCAN-5M external baseline comparison, part 2 -- the 3 models that need
# the "modern" venv (torch 2.5.1 + transformers 4.48 + mamba_ssm), set up via
# slurm/setup_env_modern.sh, because they don't work in the main
# BarcodeMAE_venv (torch 2.1.1 + transformers 4.29.2):
#   - HyenaDNA-tiny : needs newer transformers AutoConfig/trust_remote_code
#     fallback behaviour to resolve its custom config class.
#   - Caduceus-PS-1k: remote code hard-imports mamba_ssm at load time
#     (torch~=2.5.0).
#   - GENA-LM        : config.json model_type "modernbert", unsupported by
#     transformers before 4.48.0.
#
# REQUIRES slurm/setup_env_modern.sh to have been run once first.
#
# Results append to the SAME results files as bioscan5m_external_baselines.sh
# so the final table draws from one place.
#
# Submit: sbatch slurm/final_scripts/bioscan5m_external_baselines_modern.sh
# ============================================================================
#SBATCH --job-name=bioscan_external_modern
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv_modern/bin/activate"

# GENA-LM (ModernBERT) wraps its embedding layer in torch.compile internally,
# and the installed triton version doesn't match what this torch build's
# inductor backend expects (ImportError: cannot import name 'triton_key').
# We don't need the compiled path for a single forward pass -- disable
# dynamo entirely so it just runs eagerly.
export TORCHDYNAMO_DISABLE=1

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

# ── Grid (3 tasks) ──────────────────────────────────────────────────────────
MODEL_IDS=(
    "LongSafari/hyenadna-tiny-1k-seqlen"
    "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
    "AIRI-Institute/moderngena-base"
)
MODEL_CLS=(
    "causal-lm"  # HyenaDNA-tiny
    "masked-lm"  # Caduceus-PS-1k
    "auto"       # GENA-LM (ModernGENA)
)
MODEL_TAGS=(
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