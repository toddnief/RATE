#!/usr/bin/bash
#SBATCH --job-name=treatment_effects
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH	--mem=64GB
#SBATCH --mail-type=END
#SBATCH --time=5:00:00

python experiments/treatment_effects.py