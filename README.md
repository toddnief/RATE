
Note (October 15, 2024): Cleaning up this code is a work in progress. This repo will be regularly updated over the next several weeks. Please reach out if you are trying to use it and encounter problems.

# RATE (Rewrite-based Attribute Treatment Estimators)

This code runs the experiments in [RATE: Score Reward Models with Imperfect Rewrites of Rewrites.](https://arxiv.org/abs/2410.11348).

TODO: Summary of the paper and experiments

## Environment Setup

First, install the conda environment located in the root of the directory:
```
conda env create -f environment.yaml
```

If you plan to run the scripts directly, activate the environment (If you plan to use the Make commands, the Make command will manage the environment for you):
```
conda activate rate
```

### .env File

The scripts expect a ```.env``` file located in the root of the directory with three things:
- An OpenAI API key
- A project directory (for saving datasets rewritten and scored datasets - doesn't have to be the same as the location of the script)
- A permissions group (the scripts automatically update the permissions of saved files to be accessible by the permissions group - if you don't have a group you'd like to use here and you're not worried about security of the files, ```users``` is fine)
- A [SLURM partition](https://slurm.schedmd.com/quickstart.html) for Make commands to use

Here's an example ```.env``` file:
```
OPENAI_API_KEY="yOuRoPeNaIaPiKeY123"
PROJECT_DIR="path/to/where/the/data/will/save"
GROUP_NAME="users"
PARTITION="general"
```

## Experiment Structure

There are four key parts of this experiment:
1. Generating datasets
2. Scoring examples
3. Calculating treatment effects
4. Visualizing results

Each of these can be run separately--see the appropriate section for more detailed instructions.

## Datasets

The datasets we used in our experiments are:
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)
- [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)
- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [ELI5](https://facebookresearch.github.io/ELI5/index.html)

### Config Files

We use a single config file for all of the settings to run experiments. The config file is broken up into the three "phases" of the experiment and the settings for each are separate.

```yaml
smoke_test: true
rewrites:
  dataset_name: "helpsteer_helpfulness" # This is used in a factory function to import the dataset template
scoring:
  model: "armorm" # Choices: "distilbert_positive", "distilbert_negative", "deberta", "armorm", "sfairxc", "ncsoft"
  dataset_folder: "scored" # Choices: "rewrites", "scored"
  dataset_name: "imdb_length" # Note: used in filename so update to match the dataset filename below (INCLUDE CONCEPT)
  dataset_filename: "archive/imdb_length_sfairxc_scored_20240918_195038.jsonl"
effects:
  dataset_name: "imdb" # Note to make sure this matches the filename; is used to create filename
  concept: "sentiment"
  score: "armorm"
  reward_key: "ArmoRM" # Note: This is the key for the reward in the dataset
  dataset_filename: "imdb_sentiment_complete_scored_20240919_152739.jsonl"
```

### Smoke Test

If you set ```smoke_test: true``` in your config file, the experiments will run with smaller datasets to avoid wasting API calls or compute resources. Check the ```__main__``` block in each script to review the smoke test limits

### Logging

TODO

## Creating Datasets

```
make create_dataset
```

### Dataset Templates

```python
from datasets import load_dataset

helpsteer = load_dataset("nvidia/HelpSteer")
original_completions = helpsteer["train"]

dataset_template = {
    "dataset_name": "helpsteer_helpfulness",  # Note: This is used in the filename
    "original_completions": original_completions,
    "n_examples": 25000,
    "w_strings": {
        "w=1": "polite and helpful",
        "w=0": "curt and unhelpful",
    },
    "w_classifier": lambda row: False
    if row["helpfulness"] in [0, 1, 2]
    else True
    if row["helpfulness"] in [3, 4]
    else None,
    "get_reward_question": lambda row: row["prompt"],
    "get_original_completion": lambda row: row["response"],
    "rewrite_prompt": """{original_completion}\n\nPlease adjust this response so it's {w_counterfactual_string}, but change *nothing* else. Only provide the response text, not an acknowledgement of the request, the prompt or any other context.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
```

### Classifying the Ground Truth

TODO

## Scoring Datasets

Note: You can pass a config file to the Make command (which is useful for scheduling multiple jobs with different configurations via SLURM). If you don't pass a value, this defaults to ```config.yaml```.:
```
make score_dataset CONFIG=custom_config.yaml
```

### Scoring Templates

TODO

## Calculating Treatment Effects

TODO

## Visualizing Results

TODO

