
Note (October 15, 2024): Cleaning up this code is a work in progress. This repo will be regularly updated over the next several weeks. Please reach out if you are trying to use it and encounter problems.

# RATE (Rewrite-based Attribute Treatment Estimators)

This code runs the experiments in [RATE: Score Reward Models with Imperfect Rewrites of Rewrites](https://arxiv.org/abs/2410.11348).

TODO: Summary of the paper and experiments

## Environment Setup

### Conda Environment

First, install the conda environment located in the root of the directory:
```bash
conda env create -f environment.yaml
```

If you plan to run the scripts directly, activate the environment (If you plan to use the Make commands, the Make command will manage the environment for you):
```bash
conda activate rate
```

### .env File

The scripts expect a ```.env``` file located in the root of the directory with three things:
- An OpenAI API key
- A project directory (for saving datasets rewritten and scored datasets - doesn't have to be the same as the location of the script)
- A permissions group (the scripts automatically update the permissions of saved files to be accessible by the permissions group - if you don't have a group you'd like to use here and you're not worried about security of the files, ```users``` is fine)
- A [SLURM partition](https://slurm.schedmd.com/quickstart.html) for Make commands to use

Here's an example ```.env``` file:
```bash
OPENAI_API_KEY="yOuRoPeNaIaPiKeY123"
PROJECT_DIR="path/to/where/the/data/will/save"
GROUP_NAME="users"
PARTITION="general"
```

## Experiment Structure

There are four key parts of this experiment:
1. [Generating datasets](#generating-datasets)
2. [Scoring datasets](#scoring-datasets)
3. [Calculating treatment effects](#calculating-treatment-effects)
4. [Visualizing results](#visualizing-results)

Each of these can be run separately--see the appropriate section for more detailed instructions.

### Config Files

We use a single config file for all of the settings to run experiments. The config file is broken up into the three "phases" of the experiment and the settings for each are separate.

Details on the settings for each experiment are in their section, but here is an example config file for the IMDB dataset, rewriting on the concept "length" and scoring using the ArmoRM reward model.

```yaml
smoke_test: true
rewrites:
  dataset_name: "imdb_length" # This is used in a factory function to import the dataset template, must match a template in dataset_templates/
scoring:
  model: "armorm" # Choices: "distilbert_positive", "distilbert_negative", "deberta", "armorm", "sfairxc", "ncsoft"
  dataset_folder: "scored" # Choices: "rewrites", "scored"
  dataset_name: "imdb_length" # Note: used in filename so update to match the dataset filename below (INCLUDE CONCEPT)
  dataset_filename: "archive/imdb_length_sfairxc_scored_20240918_195038.jsonl"
effects:
  dataset_name: "imdb" # Note: this is used to create the filename for the calculated effects
  concept: "length"
  score: "armorm"
  reward_key: "ArmoRM" # Note: This is the key for the reward in the dataset
  dataset_filename: "imdb_sentiment_complete_scored_20240919_152739.jsonl"
```

### Smoke Test

If you set ```smoke_test: true``` in your config file, the experiments will run with smaller datasets to avoid wasting API calls or compute resources. Check the ```__main__``` block in each script to review the smoke test limits

### Logging

Logs are created in ```logs/``` when running each script with Make commands.

Logging is configured in ```experiments/constants.py```.

## Generating Datasets

To generate datasets, we go through the following process:
- Choose a dataset
- Classify the ground truth of the target concept on each example
- Make calls to the [OpenAI batch API](https://platform.openai.com/docs/guides/batch) to generate rewrites and rewrites of rewrites on the target concept

### Dataset Templates

Each dataset is managed by a dataset template that lives in ```dataset_templates```.

The dataset template specifies the following:
- How to load the baseline dataset
- How to classify the "ground truth" for each example
  - This could be a ```lambda``` function on each example in the dataset or some other form of classifier
- The "question" and the "response" for the example when scoring using a reward model
- The prompt passed to the OpenAI API to generate rewrites
  - Note: The same prompt is used for both rewrites and rewrites of rewrites

After running this script, the completed dataset and the intermediate files used for API submission will be saved to a ```data``` folder in the project directory specified in your [```.env``` file](#env-file).

#### Example Dataset Template

Here is an example dataset template.

Key things to notice:
- This dataset is available on Huggingface so is loaded directly using ```load_dataset```.
- In this example, we are using "helpfulness" as our concept (denoted by "w" in our notation) to modify - we make this more specific for rewrites by specifying W=1 as "polite and helpful" and W=0 as "curt and unhelpful"
- Each example in the HelpsSteer dataset has a human annotated score from 0-5. We define the "ground truth" value of "helpfulness" to be 1 if the response is scored 3 or 4 and 0 otherwise. We implement this by passing a ```lambda``` function as the ```w_classifier``` that extracts the human annotated value from each example.
- We also need to specify a "question" and "response" for each example so that we can score using our reward models. This dataset is structured as "prompt" and "response" so we define a ```lambda``` function to extract these.

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

### Make Command for Dataset Generation

When you have created a template for your dataset (and updated the [config yaml file](#config-files) with the appropriate settings), you can schedule this as a SLURM job using Make:

```bash
make create_dataset
```

### Datasets

The datasets we used in our experiments are:
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)
- [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)
- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [ELI5](https://facebookresearch.github.io/ELI5/index.html)

## Scoring Datasets

Note: You can pass a config file to the Make command (which is useful for scheduling multiple jobs with different configurations via SLURM). If you don't pass a value, this defaults to ```config.yaml```.:
```bash
make score_dataset CONFIG=custom_config.yaml
```

### Scoring Templates

```python
import torch
from constants import DEVICE
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, trust_remote_code=True
).to(DEVICE)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)


def _score_example(
    model,
    tokenizer,
    question,
    answer,
    device=DEVICE,
    truncation=True,
):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    with torch.no_grad():
        with autocast():
            model = model.to(device)
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            ).to(device)
            outputs = model(inputs)
            reward = outputs.score.float().item()
    del inputs, outputs  # Explicitly free up memory to prevent OOM
    return reward
```

## Calculating Treatment Effects

```bash
make treatment_effect
```

## Visualizing Results

TODO

