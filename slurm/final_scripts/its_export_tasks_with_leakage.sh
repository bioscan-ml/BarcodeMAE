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