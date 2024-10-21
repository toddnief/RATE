""""""

import argparse
import json
from pathlib import Path

import torch
import yaml
from constants import (
    DEVICE,
    FILE_ID,
    REWRITES_DIR,
    SCORED_DIR,
    SMOKE_TEST,
    load_config,
    logging,
)
from scoring_template import load_scoring_utils
from torch.cuda.amp import autocast
from utils import (
    load_dataset_from_json,
    serialize_experiment_template,
    set_group_and_permissions,
    write_to_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _score_example(
    model, tokenizer, question, answer, get_reward, device=DEVICE, truncation=True
):
    with torch.no_grad():
        with autocast():
            inputs = tokenizer(
                question,
                answer,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            ).to(device)
            outputs = model(**inputs)
    reward = get_reward(outputs)
    del inputs, outputs  # Explicitly free up memory to prevent OOM
    return reward


def score_and_save_dataset(dataset, save_path, batch_size=128, **experiment_template):
    logging.info(f"Loading: {experiment_template['reward_model_path']}")
    reward_model_name = experiment_template["reward_model_name"]
    reward_model = experiment_template["reward_model"]
    reward_tokenizer = experiment_template["reward_tokenizer"]
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    with open(save_path, "w") as f:
        # Note: Use batch processing to prevent OOM
        for i in range(0, len(dataset), batch_size):
            batch_ids = list(dataset.keys())[i : i + batch_size]
            batch_data = {}

            for id in batch_ids:
                example = dataset[id]
                dataset_entry = dataset[id]
                dataset_entry[reward_model_name] = {}

                for key, completion in example["completions"].items():
                    dataset_entry[reward_model_name][key] = experiment_template[
                        "_score_example"
                    ](
                        reward_model,
                        reward_tokenizer,
                        example["reward_question"],
                        completion,
                    )
                batch_data[id] = dataset_entry

            # Write each entry in batch_data as a separate JSONL line
            for entry in batch_data.values():
                json.dump(entry, f)
                f.write("\n")

            # Free memory after each batch
            del batch_data
            torch.cuda.empty_cache()

            logging.info(f"Scored batch {i // batch_size + 1}/{total_batches}.")

    set_group_and_permissions(save_path)

    logging.info(f"Saved scored dataset to: {save_path}")


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

    scoring_template = load_scoring_utils(config["scoring"]["model"])

    dataset_path = (
        REWRITES_DIR
        if config["scoring"]["dataset_folder"] == "rewrites"
        else SCORED_DIR
    )
    dataset_path = dataset_path / config["scoring"]["dataset_filename"]
    dataset = load_dataset_from_json(dataset_path)

    if SMOKE_TEST:
        dataset = {k: v for k, v in list(dataset.items())[:10]}

    # TODO: This is a hack to get around inconsistent formatting if reading from the rewrites - COMMENT THIS OUT FOR ELI5
    # for i, example in enumerate(dataset.values()):
    #     example["reward_question"] = "Write a movie review: "
    #     if "reward" in example:
    #         del example["reward"]

    EXPERIMENT_NAME = (
        config["scoring"]["dataset_name"] + "_" + config["scoring"]["model"]
    )
    EXPERIMENT_NAME = EXPERIMENT_NAME + "_smoke_test" if SMOKE_TEST else EXPERIMENT_NAME
    save_path = SCORED_DIR / f"{EXPERIMENT_NAME}_scored_{FILE_ID}.jsonl"
    logging.info(f"Scoring dataset and saving to: {dataset_path}")
    scored_dataset = score_and_save_dataset(dataset, save_path, **scoring_template)

    serialized_experiment_template = serialize_experiment_template(scoring_template)
    experiment_template_filename = (
        SCORED_DIR / f"{EXPERIMENT_NAME}_template_{FILE_ID}.txt"
    )
    write_to_json(serialized_experiment_template, experiment_template_filename)
