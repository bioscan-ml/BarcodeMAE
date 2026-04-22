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

# Build a skip-list from any previous run logged in res.out so we don't
# re-evaluate checkpoints that were already attempted.
SKIP_LIST=""
if [ -f res.out ]; then
    TMP_SKIP=$(mktemp)
    # Extract .pt filenames from "File:", "ERROR evaluating", and "SKIP ...:" lines.
    grep -E '^File:|ERROR evaluating|^SKIP' res.out \
        | grep -oE '[^[:space:]/]+\.pt' \
        | sort -u > "$TMP_SKIP"
    N_SKIP=$(wc -l < "$TMP_SKIP")
    echo "Parsed $N_SKIP checkpoint name(s) to skip from res.out"
    SKIP_LIST="--skip-list $TMP_SKIP"
fi

python barcodebert/eval_scan_and_test.py \
    --data-dir             ./data/ITS-5M \
    --checkpoint-base      ./model_checkpoints/ITS-5M \
    --results-dir          ./eval_results \
    --min-epochs           10 \
    --batch-size           128 \
    --cpu-workers          8 \
    $SKIP_LIST

[ -n "$TMP_SKIP" ] && rm -f "$TMP_SKIP"

echo "Done at: $(date)"