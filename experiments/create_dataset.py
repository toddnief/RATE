"""Script to create a dataset of completions and rewrites using the OpenAI API."""

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from constants import (
    API_DIR,
    FILE_ID,
    REWRITES_DATASET_NAME,
    REWRITES_DIR,
    SMOKE_TEST,
    load_config,
    logging,
)
from dotenv import load_dotenv
from gpt4_api import create_api_batch, submit_batch
from openai import OpenAI
from utils import serialize_experiment_template, write_to_json

SCRIPT_DIR = Path(__file__).resolve().parent


def write_batch_input(batch_input: List[Dict[str, Any]], filename: str) -> None:
    """
    Writes a batch of input entries to a file in JSONL (JSON Lines) format.

    Parameters:
        batch_input: A list of dictionaries representing the input data.
        filename: The name of the file where the JSONL data will be written.

    Returns:
        None
    """
    with open(filename, "w") as jsonl_file:
        for entry in batch_input:
            jsonl_file.write(json.dumps(entry) + "\n")


def add_json_to_dataset(
    dataset: Dict[str, Dict[str, Any]], completion_key: str, json_file_path: str
) -> Dict[str, Dict[str, Any]]:
    """
    Updates a dataset with completions from a JSONL file, keyed by custom IDs.

    Parameters:
        dataset: A dictionary where the keys are custom IDs and the values are dictionaries of dataset entries.
        completion_key: The key under which the completion will be added in each dataset entry.
        json_file_path: The path to the JSONL file containing the new completions.

    Returns:
        The updated dataset with completions added under the specified key.
    """
    with open(json_file_path, "r") as file:
        for line in file:
            line = json.loads(line)
            dataset_entry = dataset[line["custom_id"]]
            rewritten_completion = line["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            dataset_entry["completions"][completion_key] = rewritten_completion
            dataset[line["custom_id"]] = dataset_entry
    return dataset


def create_dataset(
    client: Any,
    file_id: str = FILE_ID,
    data_dir: Path = Path(API_DIR),
    **dataset_template: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Creates a dataset with original completions and rewrites by calling the OpenAI batch API.

    Parameters:
        client: The API client used for submitting and retrieving data.
        file_id: A string identifier for the file, used to differentiate between different runs.
        data_dir: Directory where input/output JSONL files will be written and read.
        dataset_template: A dictionary containing dataset configuration including functions to get completions,
                          reward questions, classifiers, and other dataset-related parameters.

    Returns:
        A dictionary representing the dataset, where each entry contains original completions, rewrites,
        classifier results, and reward-related information.
    """
    original_completions = dataset_template["original_completions"]
    dataset_name = dataset_template["dataset_name"]
    get_original_completion = dataset_template["get_original_completion"]
    get_reward_question = dataset_template["get_reward_question"]
    w_classifier = dataset_template["w_classifier"]
    n_examples = dataset_template["n_examples"]

    dataset = {}
    for i, example in enumerate(original_completions):
        original_completion = get_original_completion(example)
        w_original = w_classifier(example)

        # Note: Skip unclassified examples
        if w_original is None:
            continue

        # Create dataset entry
        dataset[str(i)] = {
            "w_original": w_original,
            "completions": {
                "original": original_completion,
            },
            "reward_question": get_reward_question(example),
        }

        if i >= n_examples:
            break

    # TODO: This is hacky
    rewrite_config = [
        ("first rewrite", "original", "rewrite", False),
        ("second rewrite", "rewrite", "rewritten rewrite", True),
    ]
    for (
        rewrite_string,
        og_completion_key,
        rewrite_completion_key,
        rewritten_rewrite,
    ) in rewrite_config:
        batch_input = create_api_batch(
            dataset,
            completion_key=og_completion_key,
            rewritten_rewrite=rewritten_rewrite,
            **dataset_template,
        )

        batch_input_filename = (
            data_dir / f"{dataset_name}_{rewrite_string}_input_{file_id}.jsonl"
        )
        logging.info(f"Writing {rewrite_string} batch to: {batch_input_filename}")
        write_to_json(batch_input, batch_input_filename, format="jsonl")

        content = submit_batch(
            client, batch_input_filename, file_id, **dataset_template
        )
        batch_output_filename = (
            data_dir / f"{dataset_name}_{rewrite_string}_output_{file_id}.jsonl"
        )
        # Note: Need to write file so we can retrieve the completions
        content.write_to_file(batch_output_filename)

        dataset = add_json_to_dataset(
            dataset,
            completion_key=rewrite_completion_key,
            json_file_path=batch_output_filename,
        )
    return dataset


if __name__ == "__main__":
    # Note: Use argparse to allow submission of config file via slurm
    parser = argparse.ArgumentParser(description="Scoring script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",  # Default to config.yaml in SCRIPT_DIR if not provided
        help="Path to the config file",
    )
    args = parser.parse_args()

    yaml_path = SCRIPT_DIR / args.config

    # Note: Lazy loads config constants
    load_config(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    REWRITES_DATASET_NAME = config["rewrites"]["dataset_name"]

    def load_rewrite_template(template_name):
        try:
            module = importlib.import_module(f"dataset_templates.{template_name}")
            return module.dataset_template
        except ImportError:
            raise ValueError(
                f"Template '{template_name}' could not be imported. Please check the name."
            )

    dataset_template = load_rewrite_template(REWRITES_DATASET_NAME)

    if SMOKE_TEST:
        dataset_template["n_examples"] = min(30, dataset_template["n_examples"])
        logging.info(f"Running smoke test with {dataset_template} examples...")

    # Write experiment template to file - this serializes functions as well
    logging.info("Serializing dataset template...")
    dataset_template_filename = (
        REWRITES_DIR / f"{dataset_template['dataset_name']}_template_{FILE_ID}.txt"
    )
    if SMOKE_TEST:
        dataset_template_filename = dataset_template_filename.with_stem(
            f"{dataset_template_filename.stem}_smoke_test"
        )
    serialized_dataset_template = serialize_experiment_template(dataset_template)
    write_to_json(serialized_dataset_template, dataset_template_filename)

    logging.info("Creating dataset...")
    dataset = create_dataset(
        client,
        **dataset_template,
    )

    dataset_filename = (
        REWRITES_DIR / f"{dataset_template['dataset_name']}_dataset_{FILE_ID}.json"
    )
    if SMOKE_TEST:
        dataset_filename = dataset_filename.with_stem(
            f"{dataset_filename.stem}_smoke_test"
        )
    logging.info(f"Writing completed dataset to: {dataset_filename}")
    write_to_json(dataset, dataset_filename)
