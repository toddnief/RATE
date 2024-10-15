
Note: Cleaning up this code is a work in progress. This repo will be regularly updated over the next several weeks. Please reach out if you are trying to use it and encounter problems.

# RATE (Rewrite-based Attribute Treatment Estimators)

## Environment Setup

First, install the conda environment located in the root of the directory:
```
conda env create -f environment.yaml
```

If you plan to run the scripts directly, activate the environment (If you plan to use the Make commands, the Make command will manage the environment for you):
```
conda activate rate
```

To generate rewrites, you will need an OpenAI API key saved in a ```.env``` file in the root of the directory:
```
OPENAI_API_KEY="yOuRoPeNaIaPiKeY123"
```

## Experiment Structure

There are three key parts of this experiment:
1. Generating datasets
2. Scoring examples
3. Calculating treatment effects

Each of these can be run separately--see the appropriate section for more detailed instructions.

### Config Files

We use a single config file for all of the settings to run experiments. The config file is broken up into the three "phases" of the experiment and the settings for each are separate.

### Smoke Test

### Ground Truth Datasets

## Creating Datasets

```
make create_dataset
```

### Dataset Templates

## Scoring Datasets

Note: You can pass a config file to the Make command (which is useful for scheduling multiple jobs with different configurations via SLURM). If you don't pass a value, this defaults to ```config.yaml```.:
```
make score_dataset CONFIG=custom_config.yaml
```

### Scoring Templates

## Calculating Treatment Effects

