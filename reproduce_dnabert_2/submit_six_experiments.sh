#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

jobs=(
    slurm_maelm.sh
    slurm_maelm_cls.sh
    slurm_maelm_jumbo.sh
    slurm_bert.sh
    slurm_bert_cls.sh
    slurm_bert_jumbo.sh
)

for job in "${jobs[@]}"; do
    echo "Submitting ${job}"
    sbatch "$SCRIPT_DIR/$job"
done
