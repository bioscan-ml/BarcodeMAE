#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) external baseline comparison, part 2 -- the 3 models
# that need the "modern" venv (torch 2.5.1 + transformers 4.48 + mamba_ssm),
# set up via slurm/setup_env_modern.sh, because they don't work in the main
# BarcodeMAE_venv (torch 2.1.1 + transformers 4.29.2). Same rationale as
# bioscan5m_external_baselines_modern.sh:
#   - HyenaDNA-tiny : needs newer transformers AutoConfig/trust_remote_code
#     fallback behaviour to resolve its custom config class.
#   - Caduceus-PS-1k: remote code hard-imports mamba_ssm at load time
#     (torch~=2.5.0).
#   - GENA-LM        : config.json model_type "modernbert", unsupported by
#     transformers before 4.48.0.
#
# REQUIRES slurm/setup_env_modern.sh to have been run once first, AND
# its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv).
#
# Results append to the SAME results files as its5m_external_baselines.sh
# so the final table draws from one place.
#
# Submit: sbatch slurm/final_scripts/its5m_external_baselines_modern.sh
# ============================================================================
#SBATCH --job-name=its5m_external_modern
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
export WANDB_DIR="/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
DATASET="ITS-5M"
DATA_DIR="/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1
TEMPERATURE=0.07
MAX_LEN=660

# ── Grid (3 tasks): same models as bioscan5m_external_baselines_modern.sh ────
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