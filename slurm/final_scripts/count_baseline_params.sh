#!/bin/bash
# ============================================================================
# Compute exact parameter counts for every model in Tables 5/6 (baseline
# comparison tables) plus our own BarcodeMAE+ encoder, for the manuscript's
# new "# Parameters" column. See barcodebert/count_baseline_params.py.
#
# CPU-only, no GPU needed. HF_HUB_OFFLINE=1 forces use of the local HF cache
# (the 7 HF baseline models were already downloaded when the external-baseline
# eval runs were done, so this should not need network access on the compute
# node). If any model is NOT already cached, this will fail for that model --
# rerun once from a login node first (which has internet) to populate the
# cache, then resubmit here.
#
# Adjust --mycoai-bert-ckpt / --mycoai-cnn-ckpt / --bioscan-ckpt / --its-ckpt
# below if your checkpoints live somewhere other than the defaults baked into
# count_baseline_params.py.
#
# Submit:  sbatch slurm/final_scripts/count_baseline_params.sh
# ============================================================================
#SBATCH --job-name=count_params
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=final_logs/%A/%A.out
#SBATCH --error=final_logs/%A/%A.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "final_logs/${SLURM_JOB_ID}"

python barcodebert/count_baseline_params.py

EC=$?
[ ${EC} -ne 0 ] && echo "ERROR: parameter counting failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}