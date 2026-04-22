#!/bin/bash
#SBATCH --job-name=eval_scan
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=final_logs/eval_scan/%j.out
#SBATCH --error=final_logs/eval_scan/%j.err

echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "/scratch/$USER/BarcodeMAE_venv/bin/activate"

mkdir -p final_logs/eval_scan

python barcodebert/eval_scan_and_test.py \
    --data-dir             ./data/ITS-5M \
    --checkpoint-base      ./model_checkpoints/ITS-5M \
    --results-dir          ./eval_results \
    --min-epochs           10 \
    --batch-size           128 \
    --cpu-workers          8

echo "Done at: $(date)"