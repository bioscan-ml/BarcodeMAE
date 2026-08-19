#!/bin/bash
# ============================================================================
# UNITE+INSD (ITS-5M) external baseline: HyenaDNA-tiny ONLY, for the cluster
# whose project storage is only reachable via the ~/projects/def-lila-ab/
# symlink form (no /project/6045013/ numeric mount there) -- same directory
# convention as fungi_its_knn_monitor.sh / its_aux_weight_ablation_home.sh.
#
# Single task, no array -- trimmed down from its5m_external_baselines_modern.sh's
# 3-model grid (Caduceus/GENA-LM dropped) since only HyenaDNA is needed here.
# Uses the LongSafari/hyenadna-tiny-1k-seqlen-hf repo variant (the plain
# ...-seqlen repo's config.json has no auto_map, so AutoConfig/AutoTokenizer
# can't resolve its custom code even with trust_remote_code=True -- confirmed
# via traceback on 2026-08-18; the -hf variant fixes this).
#
# REQUIRES slurm/setup_env_modern.sh to have been run once first on this
# cluster, AND its_export_tasks.sh to have been run first (produces
# data/ITS-5M/tasks/test{1,2}_tasks.csv under DATA_DIR below).
#
# 2h walltime -- a single external-model KNN pass (no pretraining) should
# comfortably fit; bump --time if it doesn't.
#
# Results append to the SAME results file as its5m_external_baselines.sh /
# its5m_external_baselines_modern.sh so the final table draws from one place.
#
# Submit: sbatch slurm/final_scripts/its5m_hyenadna_home.sh
# ============================================================================
#SBATCH --job-name=its5m_hyenadna_home
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv_modern/bin/activate"

DATASET="ITS-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/${DATASET}"
TASKS_DIR="${DATA_DIR}/tasks"
[ ! -d "${TASKS_DIR}" ] && echo "ERROR: ${TASKS_DIR} not found — run its_export_tasks.sh first" && exit 1

# Wandb/results root derived from DATA_DIR (mirrors fungi_its_knn_monitor.sh):
# DATA_DIR = <DATA_ROOT>/data/${DATASET}, so two dirname's up gets DATA_ROOT.
DATA_ROOT="$(dirname "$(dirname "${DATA_DIR}")")"

export WANDB_MODE=offline
export WANDB_DIR="${DATA_ROOT}/wandb_final/array_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
mkdir -p results_final
mkdir -p "final_logs/${SLURM_JOB_ID}"

WANDB_PROJECT="barcodemae_cls"
TEMPERATURE=0.02
MAX_LEN=660

MODEL_ID="LongSafari/hyenadna-tiny-1k-seqlen-hf"
MODEL_CLS_ARG="causal-lm"
TAG="hyenadna_tiny"

echo "Data root: ${DATA_ROOT}"
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
        --run-name                 "knn_external_${TAG}_${WEIGHTS}" \
        --results-file             results_final/KNN_ITS_external_RESULTS.txt \
        --log-wandb --wandb-project "${WANDB_PROJECT}"
    EC=$?; [ ${EC} -ne 0 ] && echo "ERROR: knn_its_clean.py failed for ${TAG}/${WEIGHTS}" && OVERALL_EXIT=${EC}
done

echo "All done at: $(date) | exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}