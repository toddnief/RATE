#!/usr/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH	--mem=64GB
#SBATCH --mail-type=END
#SBATCH --time=5:00:00

python experiments/create_dataset.py