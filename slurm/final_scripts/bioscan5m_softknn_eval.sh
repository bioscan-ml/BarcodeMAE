#!/bin/bash
# ============================================================================
# BIOSCAN-5M Soft (distance-weighted) KNN re-eval — every checkpoint that
# currently has COMPLETE uniform-vote KNN results, re-evaluated with
# --knn-weights distance for a direct soft-vs-hard comparison.
#
# Grid is hand-curated from what's actually complete in results_bioscan5m/
# as of 2026-07-20 (verified against each source script's exact checkpoint
# path pattern). 68 total tasks (configs x applicable reprs):
#   - Main sweep (bioscan5m_final.sh):            10 configs (26 w/ reprs), ALL complete
#   - Taxonomy-level family/order (..._level.sh): 6/12 configs (18 w/ reprs) complete
#       (maelm+binary, maelm+triplet, transformer+binary -- both levels;
#        missing: *+ce and transformer+triplet, still training)
#   - Taxonomy-level bin (..._level_bin.sh):       6/6 configs (18 w/ reprs), all complete
#   - Triplet mining (..._mining.sh):              2/2 configs (6 w/ reprs), both archs complete
# NOT included (not run yet): triplet margin ablation, aux-loss-weight
# ablation. Also NOT included: lambda1/lambda2 results
# (results_bioscan5m/*_lambda*.txt) -- these use relative "./checkpoints/..."
# paths from a different (non-cluster) machine, not reconstructable
# cluster-absolute paths, and are duplicate reruns of the same main
# checkpoints anyway, not new ablations. Also not included: the random-init
# baseline (uses random_knn.py, a separate script that doesn't have
# --knn-weights wired up yet).
#
# Output: results_final/KNN_distance_RESULTS_final.txt (auto-routed there by
# knn_results_path() in evaluation.py, since we pass --knn-weights distance).
#
# Submit:  sbatch slurm/final_scripts/bioscan5m_softknn_eval.sh
# ============================================================================
#SBATCH --job-name=bioscan_softknn
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-67%8
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
TAXALEVEL_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/taxonomy_level/${DATASET}"
MINING_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final/ablations/triplet_mining/${DATASET}"

K_MER=6; N_LAYERS=6; N_HEADS=6

# ── Build the (ckpt|arch|reprs|run_tag) grid ────────────────────────────────
GRID_CKPT=(); GRID_ARCH=(); GRID_REPR=(); GRID_TAG=()

add_config () {
    # One grid entry per representation available for this config.
    local ckpt="$1" arch="$2" run_tag="$3" reprs_csv="$4"
    IFS=',' read -ra reprs <<< "${reprs_csv}"
    for r in "${reprs[@]}"; do
        GRID_CKPT+=("${ckpt}"); GRID_ARCH+=("${arch}"); GRID_REPR+=("${r}"); GRID_TAG+=("${run_tag}")
    done
}

ckpt_file () { [ "$1" = "maelm" ] && echo "checkpoint_encoder.pt" || echo "checkpoint.pt"; }
run_suffix () { [ "$1" = "maelm" ] && echo "k${K_MER}_${N_LAYERS}L${N_HEADS}H_6DL6DH_$1" || echo "k${K_MER}_${N_LAYERS}L${N_HEADS}H_$1"; }

# --- Main sweep (10 configs, all complete) ---------------------------------
for ARCH in "maelm" "transformer"; do
    SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_nocls/${CFILE}"       "${ARCH}" "main_${ARCH}_nocls"       "tokens"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_none/${CFILE}"    "${ARCH}" "main_${ARCH}_cls_none"    "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_binary/${CFILE}"  "${ARCH}" "main_${ARCH}_cls_binary"  "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_triplet/${CFILE}" "${ARCH}" "main_${ARCH}_cls_triplet" "tokens,cls,tokens_with_cls"
    add_config "${MAIN_ROOT}/final_${SUFFIX}_cls_ce/${CFILE}"      "${ARCH}" "main_${ARCH}_cls_ce"      "tokens,cls,tokens_with_cls"
done

# --- Taxonomy-level family/order (6/12 complete) ---------------------------
for LEVEL in "family" "order"; do
    for ARCH_AUX in "maelm:binary" "maelm:triplet" "transformer:binary"; do
        ARCH="${ARCH_AUX%%:*}"; AUX="${ARCH_AUX##*:}"
        SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
        add_config "${TAXALEVEL_ROOT}/abl_taxa${LEVEL}_${SUFFIX}_cls_${AUX}/${CFILE}" \
            "${ARCH}" "taxa${LEVEL}_${ARCH}_${AUX}" "tokens,cls,tokens_with_cls"
    done
done

# --- Taxonomy-level bin (6/6 complete) --------------------------------------
for ARCH in "maelm" "transformer"; do
    for AUX in "binary" "triplet" "ce"; do
        SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
        add_config "${TAXALEVEL_ROOT}/abl_taxabin_${SUFFIX}_cls_${AUX}/${CFILE}" \
            "${ARCH}" "taxabin_${ARCH}_${AUX}" "tokens,cls,tokens_with_cls"
    done
done

# --- Triplet mining (2/2 complete: both archs) ------------------------------
for ARCH in "maelm" "transformer"; do
    SUFFIX="$(run_suffix "${ARCH}")"; CFILE="$(ckpt_file "${ARCH}")"
    add_config "${MINING_ROOT}/abl_miningrandom_${SUFFIX}_cls_triplet/${CFILE}" \
        "${ARCH}" "mining_${ARCH}" "tokens,cls,tokens_with_cls"
done

# Triplet margin and aux-loss-weight ablations NOT included -- not run yet.

TOTAL=${#GRID_CKPT[@]}
echo "Grid has ${TOTAL} entries (expected 68)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= grid size ${TOTAL} -- nothing to do."
    exit 0
fi

CKPT="${GRID_CKPT[$SLURM_ARRAY_TASK_ID]}"
ARCH="${GRID_ARCH[$SLURM_ARRAY_TASK_ID]}"
REPR="${GRID_REPR[$SLURM_ARRAY_TASK_ID]}"
TAG="${GRID_TAG[$SLURM_ARRAY_TASK_ID]}"

[ ! -f "${CKPT}" ] && echo "ERROR: checkpoint not found at ${CKPT}" && exit 1

RUN_NAME="softknn_${TAG}_${REPR}"
echo "Task: ${TAG} | Arch: ${ARCH} | Repr: ${REPR} | Ckpt: ${CKPT} | Run: ${RUN_NAME}"

python barcodebert/knn_probing.py \
    --pretrained-checkpoint "${CKPT}" --dataset "${DATASET}" --data-dir "${DATA_DIR}" \
    --representation_type "${REPR}" --taxon genus --n-neighbors 1 3 5 7 \
    --knn-weights distance \
    --run-name "${RUN_NAME}" \
    --results-file results_final/KNN_RESULTS_final.txt \
    --wandb-project "${WANDB_PROJECT}" --log-wandb
EC=$?

echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
