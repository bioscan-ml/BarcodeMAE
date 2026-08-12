#!/bin/bash
# ============================================================================
# Export ITS-5M leakage-free task CSVs (prerequisite for its_knn_clean_eval.sh)
#
# Runs analyze_its_overlap.py --export-dir once to produce
# tasks/test{1,2,3}_tasks.csv: id, genus, species, task, where task is one of
# species_level / genus_level / unusable / exact_duplicate /
# substring_duplicate / unknown_species. its_knn_clean_eval.sh reads these
# to restrict each test set's query specimens to the two well-posed tasks.
#
# CPU-only, no GPU needed, no model loaded — just fasta/mycoai parsing +
# pandas. Takes well under an hour even for the 5.2M-specimen train set.
#
# Submit:  sbatch its_export_tasks.sh
# ============================================================================
#SBATCH --job-name=its_export_tasks
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

DATA_DIR="/project/6045013/m4safari/BarcodeMAE/data/ITS-5M"
TASKS_DIR="${DATA_DIR}/tasks"

python barcodebert/analyze_its_overlap.py \
    --data-dir "${DATA_DIR}" \
    --export-dir "${TASKS_DIR}"

EC=$?
[ ${EC} -ne 0 ] && echo "ERROR: export failed"
echo "All done at: $(date) | exit: ${EC}"
exit ${EC}
