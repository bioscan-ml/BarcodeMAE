#!/bin/bash
# ============================================================================
# Off-the-Shelf External HuggingFace Baseline — KNN, zero-shot
#
# Evaluates published DNA foundation model checkpoints (not our own
# pretraining) via barcodebert/external_models.py's generic AutoModel
# wrapper: --external-model-id + --external-model-cls select the checkpoint
# and HF auto-class, knn_probing.py / knn_its_clean.py do the rest unchanged.
# Representation type is forced to "tokens" (universal mean-pool) by that
# code path -- there's no CLS-token variant for these, since we don't control
# their architecture.
#
# ONE-TIME SETUP (from a LOGIN node -- compute nodes have no internet):
#   1. Separate venv with a newer `transformers` (see
#      requirements-external-baselines.txt for why this must NOT be the main
#      training venv):
#        module load StdEnv/2023 cudacore/.12.6.3 python/3.11
#        python -m venv --system-site-packages /scratch/$USER/BarcodeMAE_external_venv
#        source /scratch/$USER/BarcodeMAE_external_venv/bin/activate
#        pip install --no-index -r requirements-external-baselines.txt || \
#            pip install -r requirements-external-baselines.txt
#   2. Pre-download every checkpoint in the grid below into the HF cache so
#      the (offline) compute node can find it -- compute nodes run with
#      HF_HUB_OFFLINE=1 below, so anything not already cached will fail:
#        source /scratch/$USER/BarcodeMAE_external_venv/bin/activate
#        export HF_HOME=/scratch/$USER/hf_cache
#        for m in zhihan1996/DNABERT-2-117M \
#                 zhihan1996/DNABERT-S \
#                 InstaDeepAI/nucleotide-transformer-500m-human-ref \
#                 PoetschLab/GROVER \
#                 AIRI-Institute/gena-lm-bert-base-t2t \
#                 LongSafari/hyenadna-tiny-1k-seqlen-hf \
#                 zehui127/Omni-DNA-116M; do
#            huggingface-cli download "$m" --trust-remote-code >/dev/null || huggingface-cli download "$m"
#        done
#
# VERIFIED to exist on the HF Hub with the given architecture/trust_remote_code
# expectations as of 2026-08-10 (checked via the Hub API), but NOT
# runtime-tested end-to-end through this pipeline -- smoke-test DNABERT-2/-S
# and GENA-LM interactively first (their custom modeling code has had
# transformers-version friction reported by others in the past); Nucleotide
# Transformer and GROVER use no custom code at all and are the safest bets.
#
# Caduceus (kuleshov-group/caduceus-*) is DELIBERATELY left out of the grid:
# it's a Mamba state-space model and almost certainly needs mamba-ssm +
# causal-conv1d (CUDA kernels), which are not in
# requirements-external-baselines.txt. BarcodeMamba/BarcodeMamba+ are plain
# GitHub repos, not HF AutoModel-compatible at all -- see external_models.py's
# module docstring. Add mamba-ssm/causal-conv1d to the external venv and a
# GRID_* entry here if you want Caduceus included.
#
# DATASET is an env var like the rest of slurm/final_scripts/*.sh.
#
# Submit for BIOSCAN-5M:  sbatch --export=DATASET=BIOSCAN-5M external_baseline_knn.sh
# Submit for ITS-5M:      sbatch --export=DATASET=ITS-5M     external_baseline_knn.sh
#
# Results: results_final/EXTERNAL_BASELINE_KNN_RESULTS.txt (BIOSCAN-5M)
#          results_final/EXTERNAL_BASELINE_KNN_ITS_CLEAN_RESULTS.txt (ITS-5M)
#
# Time budget: 500M-parameter models + one-sequence-at-a-time embedding (see
# random_baseline_knn.sh) is slower than either factor alone suggests. 12h
# to be safe; no resume/checkpoint here, so a timeout loses the whole pass.
# ============================================================================
#SBATCH --job-name=external_baseline_knn
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

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Dataset: ${DATASET:-BIOSCAN-5M} | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
# Separate venv (newer transformers) -- NOT the main training venv. See
# requirements-external-baselines.txt for why the two must stay isolated.
source "/scratch/$USER/BarcodeMAE_external_venv/bin/activate"

# Compute nodes have no internet access -- rely entirely on the HF cache
# populated from a login node in the one-time setup above.
export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/wandb_final/array_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_ARRAY_JOB_ID}"

nvidia-smi
python -c "import torch, transformers; print(f'PyTorch {torch.__version__} | transformers {transformers.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

# ── Grid ──────────────────────────────────────────────────────────────────────
# LABEL is just for the run name / results tag; MODEL_CLS matches
# --external-model-cls's choices (auto / masked-lm / causal-lm); MAX_LEN
# matches --external-max-length (the model's own native context, not ours --
# see external_models.py's load_external_model docstring).
GRID_LABEL=(  "dnabert2"                     "dnabert-s"              "nt500m"                                          "grover"               "gena-lm-bert-base-t2t"             "hyenadna-tiny-1k"                     "omni-dna-116m"              )
GRID_MODEL=(  "zhihan1996/DNABERT-2-117M"    "zhihan1996/DNABERT-S"   "InstaDeepAI/nucleotide-transformer-500m-human-ref" "PoetschLab/GROVER"    "AIRI-Institute/gena-lm-bert-base-t2t" "LongSafari/hyenadna-tiny-1k-seqlen-hf" "zehui127/Omni-DNA-116M"     )
GRID_CLS=(    "auto"                         "auto"                   "auto"                                            "masked-lm"            "masked-lm"                          "causal-lm"                             "causal-lm"                  )
GRID_MAXLEN=( 660                            660                      660                                               660                    660                                  1000                                    660                          )

TOTAL=${#GRID_LABEL[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= grid size ${TOTAL} — nothing to do."
    exit 0
fi

LABEL="${GRID_LABEL[$SLURM_ARRAY_TASK_ID]}"
MODEL_ID="${GRID_MODEL[$SLURM_ARRAY_TASK_ID]}"
MODEL_CLS="${GRID_CLS[$SLURM_ARRAY_TASK_ID]}"
MAX_LEN="${GRID_MAXLEN[$SLURM_ARRAY_TASK_ID]}"

# ── Dataset (overridable via --export=DATASET=...) ────────────────────────────
DATASET="${DATASET:-BIOSCAN-5M}"
if [ "${DATASET}" = "ITS-5M" ]; then
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
else
    DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
fi

RUN_NAME="external_${LABEL}"

echo "Dataset: ${DATASET} | Model: ${MODEL_ID} (${MODEL_CLS}, max_len=${MAX_LEN}) | Run: ${RUN_NAME}"

if [ "${DATASET}" = "BIOSCAN-5M" ]; then
    python barcodebert/knn_probing.py \
        --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --external-model-id "${MODEL_ID}" --external-model-cls "${MODEL_CLS}" --external-max-length ${MAX_LEN} \
        --taxon genus --n-neighbors 1 3 5 7 --metric cosine \
        --run-name "${RUN_NAME}" \
        --results-file results_final/EXTERNAL_BASELINE_KNN_RESULTS.txt
    EC=$?
else
    TASKS_DIR="${DATA_DIR}/tasks"
    [ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1
    python barcodebert/knn_its_clean.py \
        --data-dir "${DATA_DIR}" --tasks-dir "${TASKS_DIR}" \
        --external-model-id "${MODEL_ID}" --external-model-cls "${MODEL_CLS}" --external-max-length ${MAX_LEN} \
        --n-neighbors 1 3 5 7 --metric cosine \
        --run-name "${RUN_NAME}" \
        --results-file results_final/EXTERNAL_BASELINE_KNN_ITS_CLEAN_RESULTS.txt
    EC=$?
fi

[ ${EC} -ne 0 ] && echo "ERROR: external baseline KNN failed"

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}