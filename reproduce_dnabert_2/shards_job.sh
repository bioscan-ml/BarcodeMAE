#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --mem-per-cpu=16000M      # memory; default unit is megabytes
#SBATCH --time=0-6:00:00
#SBATCH --cpus-per-task=4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.11
source /home/pmillana/xformers/bin/activate
cd /home/pmillana/projects/def-lila-ab/pmillana/reproduce_dnabert_2
python write_shards.py
