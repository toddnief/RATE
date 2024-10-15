CONFIG ?= config.yaml  # Default config file if CONFIG is not provided

_conda = conda
SBATCH = $(_conda) run -n ${ENV_NAME} sbatch
SRUN = $(_conda) run -n ${ENV_NAME} srun

# TODO: set these in .env
ENV_NAME = editeval
PARTITION = general
PROJECT_DIR = /net/projects/veitch/prompt_distributions/
DATA_DIR = $(PROJECT_DIR)data/
RESULTS_DIR = $(PROJECT_DIR)results/
REWRITES_DIR = $(DATA_DIR)rewrites/
GROUP_NAME = veitch-lab
PERMISSIONS = 664

logs_dir = ${workdir}logs/
DATE := $(shell date +"%Y%m%d_%H%M%S")
LOG_FILE_PREFIX = ${logs_dir}${DATE}
output_file = ${LOG_FILE_PREFIX}_res.txt
err_file = ${LOG_FILE_PREFIX}_err.txt

workdir = ./

.PHONY: create_dataset
create_dataset:
	${SBATCH} \
		--partition=$(PARTITION) \
		--output="$(output_file)" \
		--error="$(err_file)" \
		$(workdir)create_dataset.slurm

# Usage: make score_dataset CONFIG=config.yaml
.PHONY: score_dataset
score_dataset:
	${SBATCH} \
		--partition=$(PARTITION) \
		--output="$(output_file)" \
		--error="$(err_file)" \
		--export=ALL,CONFIG=$(CONFIG) \
		$(workdir)score_dataset.slurm

.PHONY: treatment_effect
treatment_effect:
	${SBATCH} \
		--partition=$(PARTITION) \
		--output="$(output_file)" \
		--error="$(err_file)" \
		$(workdir)treatment_effect.slurm