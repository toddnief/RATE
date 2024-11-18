#!/usr/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH	--mem=64GB
#SBATCH --mail-type=END
#SBATCH --time=5:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config.yaml}

python src/rate/create_dataset.py --config $CONFIG_FILE