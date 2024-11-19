
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

Create a ```.env``` file located in the root of the directory with three things:
- An OpenAI API key
- A project directory (for saving datasets, rewrites, and scored datasets - doesn't have to be the same as the location of the script)
- A permissions group (the scripts automatically update the permissions of saved files to be accessible by the permissions group - if you don't have a group you'd like to use here and you're not worried about security of the files, ```users``` is fine)
- A [SLURM partition](https://slurm.schedmd.com/quickstart.html) for Make commands to use

Here's an example ```.env``` file:
```bash
OPENAI_API_KEY=yOuRoPeNaIaPiKeY123
PROJECT_DIR=path/to/project/directory
GROUP_NAME=users
PARTITION=general
```

### File Locations and Filenames

Here is the expected file structure for the experiments (note that these directories will be created automatically in the project directory specified in the ```.env``` file):
```
project_dir/
├── data/                      # Data directory specified in .env
│  ├── batch_api/              # API artifacts stored here
│  ├── rewrites/               # Rewritten datasets stored here
│  ├── scored/                 # Scored datasets stored here
│  ├── dataset_1/              # Raw dataset folders (e.g. ELI5, hh-rlhf, etc.)
│  └── dataset_2/
└── effects/                   # Calculated treatment effects stored here
```

The scripts will append the timestamp from when the script was run to each filename as a unique identifier (e.g. ```helpsteer_complexity_dataset_20240921_213510.json```) and will also save a template file that shows the configuration settings for the run (e.g. ```helpsteer_complexity_template_20240921_213510.txt```). **Make sure to update ```config.yaml``` to use the correct file for the experiment that you plan to run!***

## Experiment Structure

There are four key parts of this experiment:
1. [Generating datasets](#generating-datasets)
2. [Scoring datasets](#scoring-datasets)
3. [Calculating treatment effects](#calculating-treatment-effects)
4. [Visualizing results](#visualizing-results)

Each of these can be run separately--see the appropriate section for more detailed instructions.

### Config Files

We use a single config file for all of the settings to run experiments. The default config file is located at ```experiments/config.yaml```. (Note: You can make additional config files for different experiment setups, which is necessary if you plan to schedule multiple experiments via SLURM).

The config file is broken up into the three "phases" of the experiment and the settings for each are separate.

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
- In this example, we are using "length" as our concept (denoted by "w" in our notation) to modify
- We define the ground truth value of "long" or "short" based on the median number of characters in our response — "length" is 1 if the response is longer than the median response and 0 if it is less than or equal to the median length
- We also need to specify a "question" and "response" for each example so that we can score using our reward models. Since the IMDB dataset doesn't have a prompt for each example, we define this to be "Write a movie review" for each example.

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

dataset_template = {
    "dataset_name": "imdb_length",
    "original_completions": dataset["train"],
    "w_strings": {
        "w=1": "longer",
        "w=0": "shorter",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: len(x) > 970,
    "get_original_completion": lambda x: x["text"],
    "reward_question": lambda x: "Write a movie review: ",  # Note: The reward models need a prompt to score completions
    "rewrite_prompt": """{original_completion}\n\n[Adjust this review so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
```

### Make Command for Dataset Generation

> **Caution:** Ensure that no Conda environment (other than the default `base`) is active before running Make commands. If you have another environment active, deactivate it with:
>
> ```bash
> conda deactivate
> ```

The Makefile will automatically manage the Conda environment for you during the job execution.

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

To score datasets, we need to update the ```config.yaml``` file and make sure we've created a [scoring template](#scoring-templates) for our reward model.

These are the relevant fields in the yaml file. Make sure the ```model``` field aligns with the name of the scoring template. ```dataset_folder``` and ```dataset_filename``` specify where the saved dataset is that you want to score. ```dataset_name``` is used when creating the scored output file.

```yaml
scoring:
  model: "armorm" # Choices: "distilbert_positive", "distilbert_negative", "armorm", "sfairxc", "ncsoft"
  dataset_folder: "rewrites" # Choices: "rewrites", "scored"
  dataset_filename: "imdb_length_dataset_20241028_142301_smoke_test.json"
  dataset_name: "imdb_length" # Note: used in output filename so update to match the dataset_name below (INCLUDE CONCEPT)
```

Note: when scoring a dataset for the first time, use ```dataset_folder: "rewrites"```. If you would like to score additional reward models on this same data, you will need to change to ```dataset_folder: "scored"``` in order to append to existing scored data. In either case, ```dataset_filename``` is the name of the file in said directory which should be scored.

### Scoring Templates

We define a scoring template for each reward model in ```scoring_templates/```. ```reward_model```, ```reward_tokenizer``` and ```_score_example``` are imported via a factory function. 

Make sure that the model name in your ```config.yaml``` and the file name in ```scoring_templates/``` match.

This is an example scoring template for the ArmoRM model.

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

### Make Command for Scoring Examples
There is a make command that will allow you to schedule a scoring job using SLURM.

Note: You can pass a config file to the Make command (which is useful for scheduling multiple jobs with different configurations via SLURM). If you don't pass a value, this defaults to ```config.yaml```.:
```bash
make score_dataset CONFIG=custom_config.yaml
```

## Calculating Treatment Effects

After scoring a dataset, we calculate the average treatment effect using our rewritten counterfactual examples.

This defaults to calculating the effect size between the rewritten rewrite and the rewrite.

Here is an example setup in ```config.yaml``` — specify the key for the saved reward in the dataset and the dataset_filename.

```yaml
effects:
  dataset_name: "imdb_length" # Note: this is used to create the filename for the calculated effects
  dataset_folder: null # null to load directly from the scored dataset_folder. Or, specifiy a subdirectory
  concept: "length"
  score: "armorm"
  reward_key: "RLHFlow/ArmoRM-Llama3-8B-v0.1" # Note: This is the key for the reward in the dataset
  dataset_filename: "imdb_length_armorm_smoke_test_scored_20241104_111220.jsonl"
```

### Make Command for Calculating Treatment Effects

You can use a make command for calculating treatment effects on SLURM-based systems.

```bash
make treatment_effects
```

## Visualizing Results

We provide various plotting functions in ```utils_plotting.py```. You can see these functions called in ```results.ipynb```.

To visualize your results, you will just need to modify ```results.ipynb``` to replace the arguments to the ```load_dataset_from_json``` calls.
```python
config = load_dataset_from_json(Path("config_plotting.jsonl"))
```
You will need to create your own ```.jsonl``` config files with your experiment templates. Here is the format:
```json
{"dataset_name": "IMDB", "concept": "Sentiment", "score": "FsfairX-LLaMA3-RM-v0.1", "reward_key": "sfairXC/FsfairX-LLaMA3-RM-v0.1", "dataset_filename": "imdb_sentiment_complete_scored_20240919_152739.jsonl"}
{"dataset_name": "HelpSteer", "concept": "Complexity", "score": "ArmoRM", "reward_key": "RLHFlow/ArmoRM-Llama3-8B-v0.1", "dataset_filename": "helpsteer_complexity_complete_scored.jsonl"}
```

## Synthetic Experiments

TODO
