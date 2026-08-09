#!/bin/bash
# ============================================================================
# BIOSCAN-5M UNIFORM (plain majority) vote KNN eval, run fresh on the
# current cluster at the SAME extended k-range as the softmax/distance runs
# (1,3,5,7,10,15,20,25,50).
#
# This exists because the previously-used "uniform" reference numbers were
# fragmented across a stale local results_bioscan5m/KNN_RESULTS_final.txt,
# an older non-cluster-machine run (BIOSCAN5M_results/, lambda1/lambda2
# files), and an incomplete mining-ablation file -- see conversation
# history. Rerunning uniform fresh here, on the same grid/checkpoints/
# k-range as bioscan5m_softmaxknn_eval.sh, gives one clean, directly-
# comparable, same-environment baseline instead of stitching old sources.
#
# Identical grid to bioscan5m_softmaxknn_eval.sh (same 35-task grid), just
# omits --knn-weights (uniform is knn_vote()'s default) and --temperature.
#
# Grid (35 total array tasks):
#   - Main sweep (bioscan5m_final.sh):              10 configs -> 26 w/ reprs
#     (2 arch x 5 objectives: nocls/cls_none/cls_binary/cls_triplet/cls_ce;
#      nocls=1 repr, others=3 reprs each)
#   - Triplet mining ablation:                        2 configs -> 6 w/ reprs
#   - Random-init baseline (random_knn.py):           3 configs, transformer
#     only (nocls/tokens, cls/cls, cls/tokens_with_cls).
#
# Output: results_final/KNN_RESULTS_final.txt (main+mining, via
# knn_probing.py, tagged "uniformknn_*" to stay distinct from any older
# entries in the same file) and results_final/RANDOM_KNN_RESULTS.txt
# (random baseline, via random_knn.py, tagged "uniformknn_random_*").
#
# Submit:  sbatch slurm/final_scripts/bioscan5m_uniformknn_eval.sh
# ============================================================================
#SBATCH --job-name=bioscan_uniformknn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-34%6
#SBATCH --output=final_logs/%A/%A_%a.out
#SBATCH --error=final_logs/%A/%A_%a.err

echo "Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | $(date)"

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

DATASET="BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/${DATASET}"
WANDB_PROJECT="barcodemae_cls"

MAIN_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/${DATASET}"
MINING_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/triplet_mining/${DATASET}"

K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Build the (ckpt|arch|reprs|run_tag) grid ────────────────────────────────
# CKPT="RANDOM" is the sentinel for the random-init baseline.
GRID_CKPT=(); GRID_ARCH=(); GRID_REPR=(); GRID_TAG=(); GRID_USECLS=()

add_config () {
    local ckpt="$1" arch="$2" run_tag="$3" reprs_csv="$4"
    IFS=',' read -ra reprs <<< "${reprs_csv}"
    for r in "${reprs[@]}"; do
        GRID_CKPT+=("${ckpt}"); GRID_ARCH+=("${arch}"); GRID_REPR+=("${r}"); GRID_TAG+=("${run_tag}"); GRID_USECLS+=("0")
    done
}

add_random () {
    # $5 = "1" if --use-cls-token should be passed (repr requires CLS)
    GRID_CKPT+=("RANDOM"); GRID_ARCH+=("$1"); GRID_REPR+=("$2"); GRID_TAG+=("$3"); GRID_USECLS+=("$4")
}

ckpt_file () { [ "$1" = "maelm" ] && echo "checkpoint_encoder.pt" || echo "checkpoint.pt"; }
run_suffix () { [ "$1" = "maelm" ] && echo "k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_$1" || echo "k${K_MER}_${N_LAYERS}L${N_HEADS}H_$1"; }

# --- Main sweep (10 configs -> 26 w/ reprs) ---------------------------------
for ARCH in "maelm" "transformer"; do
    SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_nocls/${CFILE}"       "${ARCH}" "main_${ARCH}_nocls"       "tokens"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_none/${CFILE}"    "${ARCH}" "main_${ARCH}_cls_none"    "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_binary/${CFILE}"  "${ARCH}" "main_${ARCH}_cls_binary"  "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_triplet/${CFILE}" "${ARCH}" "main_${ARCH}_cls_triplet" "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_ce/${CFILE}"      "${ARCH}" "main_${ARCH}_cls_ce"      "tokens,cls,tokens_with_cls"
done

# --- Triplet mining ablation (2 configs -> 6 w/ reprs) ----------------------
for ARCH in "maelm" "transformer"; do
    SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
    add_config "${MINING_ROOT}/abl_miningrandom_${SUFFIX}_cls_triplet/${CFILE}" \
        "${ARCH}" "mining_${ARCH}" "tokens,cls,tokens_with_cls"
done

# --- Random-init baseline (3 configs, transformer only) ---------------------
add_random "transformer" "tokens"          "random_transformer" "0"
add_random "transformer" "cls"             "random_transformer" "1"
add_random "transformer" "tokens_with_cls" "random_transformer" "1"

TOTAL=${#GRID_CKPT[@]}
echo "Grid has ${TOTAL} entries (expected 35)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= grid size ${TOTAL} -- nothing to do."
    exit 0
fi

CKPT="${GRID_CKPT[$SLURM_ARRAY_TASK_ID]}"
ARCH="${GRID_ARCH[$SLURM_ARRAY_TASK_ID]}"
REPR="${GRID_REPR[$SLURM_ARRAY_TASK_ID]}"
TAG="${GRID_TAG[$SLURM_ARRAY_TASK_ID]}"
USECLS="${GRID_USECLS[$SLURM_ARRAY_TASK_ID]}"

if [ "${CKPT}" = "RANDOM" ]; then
    RUN_NAME="uniformknn_${TAG}_${REPR}"
    USE_CLS_ARGS=()
    [ "${USECLS}" = "1" ] && USE_CLS_ARGS=(--use-cls-token)
    echo "Task: RANDOM baseline | ${TAG} | Arch: ${ARCH} | Repr: ${REPR} | Run: ${RUN_NAME}"
    python barcodebert/random_knn.py \
        --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --arch "${ARCH}" --k-mer ${K_MER} --stride ${K_MER} --n-layers ${N_LAYERS} --n-heads ${N_HEADS} \
        --encoder-embed-dim 768 "${USE_CLS_ARGS[@]}" \
        --representation-type "${REPR}" --taxon genus --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine \
        --run-name "${RUN_NAME}" --results-file results_final/RANDOM_KNN_RESULTS.txt
    EC=$?
else
    [ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1
    RUN_NAME="uniformknn_${TAG}_${REPR}"
    echo "Task: ${TAG} | Arch: ${ARCH} | Repr: ${REPR} | Ckpt: ${CKPT} | Run: ${RUN_NAME}"
    python barcodebert/knn_probing.py \
        --pretrained-checkpoint "${CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
        --representation_type "${REPR}" --taxon genus --n-neighbors 1 3 5 7 10 15 20 25 50 --metric cosine \
        --run-name "${RUN_NAME}" \
        --results-file results_final/KNN_RESULTS_final.txt \
        --wandb-project "${WANDB_PROJECT}" --log-wandb
    EC=$?
fi

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}