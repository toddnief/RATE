#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=promptdistributions
#SBATCH --output=/net/projects/veitch/prompt_distributions/logs/gpt4_%A_%a.out
#SBATCH --error=/net/projects/veitch/prompt_distributions/logs/gpt4_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


# Activate your environment, e.g., conda or virtualenv
# For Conda:
source /net/projects/veitch/reber/miniconda3/etc/profile.d/conda.sh
conda activate editeval

# Constants for parameters
MODEL_ID="gpt-4-1106-preview"
SUBSET_SIZE=4
SEED=123
NUM_TOKENS=40
NUM_NEXT_TOKENS=1
TEMPERATURE=0.7

python prompt_completions_gpt4.py --model_id $MODEL_ID --subset_size $SUBSET_SIZE --seed $SEED --num_tokens $NUM_TOKENS --num_next_tokens $NUM_NEXT_TOKENS --temperature $TEMPERATURE --job_id $SLURM_JOB_ID

conda deactivate