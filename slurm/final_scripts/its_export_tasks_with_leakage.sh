#!/bin/bash
# ============================================================================
# Export ITS-5M task CSVs WITH leakage included (counterpart to
# its_export_tasks.sh) -- same species_level/genus_level task definitions,
# but exact-duplicate and substring-duplicate specimens are NOT excluded from
# the query pools. Lets every downstream KNN eval script (knn_its_clean.py,
# knn_its_mycoai.py, knn_its_barcodemamba.py) run completely unchanged, just
# pointed at this directory instead of tasks/ -- producing the
# leakage-INCLUDED counterpart of the same baseline table.
#
# Runs analyze_its_overlap.py --export-dir --include-leaked.
#
# CPU-only, no GPU needed, no model loaded — just fasta/mycoai parsing +
# pandas. Takes well under an hour even for the 5.2M-specimen train set.
#
# Submit:  sbatch its_export_tasks_with_leakage.sh
# ============================================================================
#SBATCH --job-name=its_export_tasks_with_leakage
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=final_logs/%A/%A.out
#SBATCH --error=final_logs/%A/%A.err

echo "Job $SLURM_JOB_ID | $(date)"

module load StdEnv/2023
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

# mycoai's __init__.py calls wandb.login('allow') unconditionally on import,
# even though this script never logs anything to wandb itself. Neither
# WANDB_MODE nor TMPDIR fixed the ServiceStartTimeoutError -- wandb's default
# "service" mode forks a SEPARATE SUBPROCESS to manage state, and on this
# cluster that subprocess can't reliably see the tempdir its parent just
# created (classic HPC fork/mount-namespace flakiness), so the port-file
# write fails no matter where TMPDIR points. WANDB_START_METHOD=thread runs
# the service as a thread in the same process instead of forking a
# subprocess, which sidesteps the cross-process port file entirely.
export WANDB_MODE=disabled
export WANDB_START_METHOD=thread

mkdir -p "final_logs/${SLURM_JOB_ID}"

DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks_with_leakage"

python barcodebert/analyze_its_overlap.py \
    --data-dir "${DATA_DIR}" \
    --export-dir "${TASKS_DIR}" \
    --include-leaked

EC=$?
[ ${EC} -ne 0 ] && echo "ERROR: export failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}