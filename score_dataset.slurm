#!/usr/bin/bash
#SBATCH --job-name=score_dataset
#SBATCH	--gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH	--mem=64GB
#SBATCH --mail-type=END
#SBATCH --constraint="a100|h100"
#SBATCH --time=5:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config.yaml}

python experiments/score_dataset.py --config $CONFIG_FILE