"""Functions for calculating treatment effects from a scored dataset."""

import argparse
import pprint
from typing import Any, Dict

import numpy as np
import yaml
from constants import (
    EFFECTS_DIR,
    FILE_ID,
    ROOT_DIR,
    SCORED_DIR,
    logging,
)
from utils import load_dataset_from_json, write_to_json


def calculate_rewrite_effect(
    dataset: dict,
    original: str = "original",
    rewrite: str = "rewritten rewrite",
    **effects_template,
) -> dict:
    """
    Calculate the rewrite effect by comparing the reward model's scores between original and rewritten examples.

    Args:
        dataset: A dictionary of examples, where each example contains a reward model score for both the original
                 and rewritten versions.
        original: The key for the original text
        rewrite: The key for the rewritten text
        **effects_template: Additional keyword arguments, including "reward_key" for the reward model name.

    Returns:
        A dictionary containing:
        - The mean and standard error of the rewrite effect for examples where "w_original" is True (w_1).
        - The mean and standard error of the rewrite effect for examples where "w_original" is False (w_0).
    """
    reward_model_name = effects_template.get("reward_key", "reward")

    w_1 = []
    w_0 = []

    for example in dataset.values():
        if example["w_original"]:
            w_1.append(
                example[reward_model_name][rewrite]
                - example[reward_model_name][original]
            )
        else:
            w_0.append(
                example[reward_model_name][rewrite]
                - example[reward_model_name][original]
            )

    w_1, w_0 = np.asarray(w_1), np.asarray(w_0)

    return {
        "rewrite_effect_w_1": np.mean(w_1),
        "rewrite_effect_w_1_stderr": np.std(w_1, ddof=1) / np.sqrt(len(w_1)),
        "rewrite_effect_w_0": np.mean(w_0),
        "rewrite_effect_w_0_stderr": np.std(w_0, ddof=1) / np.sqrt(len(w_0)),
    }


def _calculate_treatment_effects(treated, untreated, paired=False):
    """
    Calculates the treatment effect and standard error for unpaired data.

    Args:
        treated: An array of outcomes for the treated group.
        untreated: An array of outcomes for the untreated group.

    Returns:
        A tuple containing the treatment effect (difference in means) and the standard error of the difference.
    """
    if not isinstance(treated, np.ndarray):
        treated = np.array(treated)
    if not isinstance(untreated, np.ndarray):
        untreated = np.array(untreated)

    treated_effect = np.mean(treated) - np.mean(untreated)
    if not paired:
        stderr = np.sqrt(
            np.var(treated, ddof=1) / len(treated)
            + np.var(untreated, ddof=1) / len(untreated)
        )
    else:
        stderr = np.std(treated - untreated, ddof=1) / np.sqrt(len(treated))
    return treated_effect, stderr


def treatment_effects_pipeline(
    dataset: Dict[str, Any], **effects_template
) -> Dict[str, float]:
    """
    Calculate treatment effects (ATE, ATT, ATU) with optional covariate adjustment and bias correction

    Args:
        dataset: A dictionary containing the dataset, where each entry is another dictionary with reward values.
        **effects_template: Optional keyword arguments to specify the keys for original and rewrite conditions,
                            as well as the reward model (e.g., "reward_key").

    Returns:
        A dictionary containing treatment effects and standard errors
    """
    w_original_key = effects_template.get("w_original_key", "original")
    w_counterfactual_key = effects_template.get("w_counterfactual_key", "rewrite")
    w_rewritten_rewrite_key = effects_template.get(
        "w_rewritten_rewrite_key", "rewritten rewrite"
    )
    reward_key = effects_template.get("reward_key", "reward")

    Y1_count = 0
    # Used for naive estimator
    Y1_rewards = []
    Y0_rewards = []
    # Used for ATT and ATU
    Y1_rewritten_rewards = []
    Y0_rewritten_rewards = []
    # Used for ATT and ATU with rewritten rewrites
    Y1_rewritten_rewrite_rewards = []
    Y0_rewritten_rewrite_rewards = []
    # Used for ATE with single rewrite
    Y_do1_rewards = []
    Y_do0_rewards = []
    # Used for ATE with rewritten rewrites
    Y_do1_rewritten_rewrite_rewards = []
    Y_do0_rewritten_rewrite_rewards = []

    for example in dataset.values():
        if example["w_original"]:
            Y1_count += 1
            Y1_rewards.append(example[reward_key][w_original_key])

            # Single rewrite
            Y_do1_rewards.append(example[reward_key][w_original_key])
            Y_do0_rewards.append(example[reward_key][w_counterfactual_key])
            Y0_rewritten_rewards.append(example[reward_key][w_counterfactual_key])

            # Rewritten rewrites
            Y_do1_rewritten_rewrite_rewards.append(
                example[reward_key][w_rewritten_rewrite_key]
            )
            Y_do0_rewritten_rewrite_rewards.append(
                example[reward_key][w_counterfactual_key]
            )
            Y1_rewritten_rewrite_rewards.append(
                example[reward_key][w_rewritten_rewrite_key]
            )
        else:
            Y0_rewards.append(example[effects_template["reward_key"]][w_original_key])

            # Single rewrite
            Y_do1_rewards.append(example[reward_key][w_counterfactual_key])
            Y_do0_rewards.append(example[reward_key][w_original_key])
            Y1_rewritten_rewards.append(example[reward_key][w_counterfactual_key])

            # Rewritten rewrites
            Y_do1_rewritten_rewrite_rewards.append(
                example[reward_key][w_counterfactual_key]
            )
            Y_do0_rewritten_rewrite_rewards.append(
                example[reward_key][w_rewritten_rewrite_key]
            )
            Y0_rewritten_rewrite_rewards.append(
                example[reward_key][w_rewritten_rewrite_key]
            )

    Y0_count = len(dataset) - Y1_count

    # Calculate pooled ssd for effect size calculation
    std_Y1 = np.std(Y1_rewards, ddof=1) if Y1_rewards else 0
    std_Y0 = np.std(Y0_rewards, ddof=1) if Y0_rewards else 0
    pooled_std = np.sqrt(
        ((Y1_count - 1) * std_Y1**2 + (Y0_count - 1) * std_Y0**2)
        / (Y1_count + Y0_count - 2)
    )

    logging.info(f"Number of w=1 examples: {Y1_count}")
    logging.info(f"Number of w=0 examples: {Y0_count}")

    # Naive effects
    naive_effect, naive_effect_stderr = _calculate_treatment_effects(
        Y1_rewards, Y0_rewards
    )

    # Single rewrite effects
    ATE_single_rewrite, ATE_single_rewrite_stderr = _calculate_treatment_effects(
        Y_do1_rewards, Y_do0_rewards, paired=True
    )
    # TODO: double check this
    ATT_single_rewrite, ATT_single_rewrite_stderr = _calculate_treatment_effects(
        Y1_rewards, Y0_rewritten_rewards, paired=True
    )
    ATU_single_rewrite, ATU_single_rewrite_stderr = _calculate_treatment_effects(
        Y1_rewritten_rewards, Y0_rewards, paired=True
    )

    # Rewritten rewrite effects
    ATE_rewritten_rewrite, ATE_rewritten_rewrite_stderr = _calculate_treatment_effects(
        Y_do1_rewritten_rewrite_rewards, Y_do0_rewritten_rewrite_rewards
    )
    ATT_rewritten_rewrite, ATT_rewritten_rewrite_stderr = _calculate_treatment_effects(
        Y1_rewritten_rewrite_rewards, Y0_rewritten_rewards
    )
    ATU_rewritten_rewrite, ATU_rewritten_rewrite_stderr = _calculate_treatment_effects(
        Y1_rewritten_rewards, Y0_rewritten_rewrite_rewards
    )

    effects = {
        "naive_effect": naive_effect,
        "naive_effect_stderr": naive_effect_stderr,
        "ATE_single_rewrite": ATE_single_rewrite,
        "ATE_single_rewrite_stderr": ATE_single_rewrite_stderr,
        "ATT_single_rewrite": ATT_single_rewrite,
        "ATT_single_rewrite_stderr": ATT_single_rewrite_stderr,
        "ATU_single_rewrite": ATU_single_rewrite,
        "ATU_single_rewrite_stderr": ATU_single_rewrite_stderr,
        "ATE_rewritten_rewrite": ATE_rewritten_rewrite,
        "ATE_rewritten_rewrite_stderr": ATE_rewritten_rewrite_stderr,
        "ATT_rewritten_rewrite": ATT_rewritten_rewrite,
        "ATT_rewritten_rewrite_stderr": ATT_rewritten_rewrite_stderr,
        "ATU_rewritten_rewrite": ATU_rewritten_rewrite,
        "ATU_rewritten_rewrite_stderr": ATU_rewritten_rewrite_stderr,
        "reward_std": pooled_std,
        "Y1_count": Y1_count,
        "Y0_count": Y0_count,
    }

    pp = pprint.PrettyPrinter(indent=4)
    logging.info(f"Treatment effects: {pp.pformat(effects)}")
    logging.info(
        f"Calculated ATE (from ATT and ATU): {(Y1_count * effects['ATT_rewritten_rewrite'] + Y0_count * effects['ATU_rewritten_rewrite'])/len(dataset)}"
    )

    rewrite_effects = calculate_rewrite_effect(dataset, **effects_template)
    logging.info(f"Rewrite effects: {pp.pformat(rewrite_effects)}")
    return effects


if __name__ == "__main__":
    # Note: Use argparse to allow submission of config file via slurm
    parser = argparse.ArgumentParser(description="Scoring script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",  # Default to config.yaml in ROOT_DIR if not provided
        help="Path to the config file",
    )
    args = parser.parse_args()

    yaml_path = ROOT_DIR / args.config
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    effects_template = config["effects"]

    EXPERIMENT_NAME = (
        effects_template["dataset_name"]
        + "_"
        + effects_template["concept"]
        + "_"
        + effects_template["score"]
    )

    logging.info(f"Running treatment effects for: {EXPERIMENT_NAME}")

    dataset_filename = effects_template["dataset_filename"]
    dataset_folder = effects_template["dataset_folder"]
    dataset_path = (
        SCORED_DIR / dataset_folder / dataset_filename
        if dataset_folder is not None
        else SCORED_DIR / dataset_filename
    )
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset_from_json(dataset_path)

    logging.info("Dataset loaded. Calculating treatment effects...")
    # effects = calculate_treatment_effects(dataset, **effects_template)
    effects = treatment_effects_pipeline(dataset, **effects_template)

    effects.update(effects_template)

    results_filename = (
        EFFECTS_DIR / f"{EXPERIMENT_NAME}_treatment_effects_{FILE_ID}.json"
    )
    logging.info(f"Saving treatment effects to: {results_filename}")
    write_to_json(effects, results_filename)
