"""Functions for calculating treatment effects from a scored dataset."""

import pprint
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from constants import (  # noqa
    EFFECTS_DIR,
    FILE_ID,
    REWRITES_DATASET_NAME,
    REWRITES_DIR,
    SCORED_DIR,
    logging,
)
from utils import load_dataset_from_json, write_to_json

SCRIPT_DIR = Path(__file__).resolve().parent


def calculate_unpaired_treatment_effect(
    treated: np.ndarray, untreated: np.ndarray
) -> Tuple[float, float]:
    """
    Calculates the treatment effect and standard error for unpaired data.

    Args:
        treated: An array of outcomes for the treated group.
        untreated: An array of outcomes for the untreated group.

    Returns:
        A tuple containing the treatment effect (difference in means) and the standard error of the difference.
    """
    treated_effect = np.mean(treated) - np.mean(untreated)
    stderr = np.sqrt(
        np.var(treated, ddof=1) / len(treated)
        + np.var(untreated, ddof=1) / len(untreated)
    )
    return treated_effect, stderr


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


def calculate_average_treatment_effects(
    dataset: Dict[str, Any], **effects_template
) -> Dict[str, float]:
    """
    Calculate the average treatment effects (ATE, ATT, ATU) and their standard errors.

    This function computes:
    - ATE (Average Treatment Effect)
    - ATT (Average Treatment Effect on the Treated)
    - ATU (Average Treatment Effect on the Untreated)
    - A naive effect
    It uses both paired and unpaired treatment effect calculations, with standard errors for each estimate.

    Args:
        dataset: A dictionary where each key is an identifier, and the value is another dictionary
                 containing outcome data for different reward models (keys for "original" and "rewrite" outcomes).
        **effects_template: Optional arguments that specify the reward model's keys.
                            Default keys:
                            - reward_key: "reward"
                            - original: "rewritten rewrite"
                            - rewrite: "rewrite"

    Returns:
        A dictionary containing treatment effects and standard errors
    """
    # TODO: This is confusing and should be cleaned up
    # The default should actually be between the rewritten rewrite and the rewrite
    reward_key = effects_template.get("reward_key", "reward")
    original = effects_template.get("original", "rewritten rewrite")
    rewrite = effects_template.get("rewrite", "rewrite")

    do_w_1 = []
    do_w_0 = []
    w_1 = []
    w_0 = []
    w_0_for_w_1 = []  # Used to calculate ATT & ATU
    w_1_for_w_0 = []

    ate_effects = []

    for example in dataset.values():
        if example["w_original"]:
            do_w_1.append(example[reward_key][original])
            do_w_0.append(example[reward_key][rewrite])

            w_1.append(example[reward_key][original])
            w_0_for_w_1.append(example[reward_key][rewrite])
            ate_effects.append(
                example[reward_key][original] - example[reward_key][rewrite]
            )
        else:
            do_w_1.append(example[reward_key][rewrite])
            do_w_0.append(example[reward_key][original])

            w_0.append(example[reward_key][original])
            w_1_for_w_0.append(example[reward_key][rewrite])
            ate_effects.append(
                example[reward_key][rewrite] - example[reward_key][original]
            )

    do_w_1, do_w_0 = np.asarray(do_w_1), np.asarray(do_w_0)
    w_1, w_0 = np.asarray(w_1), np.asarray(w_0)
    w_1_for_w_0, w_0_for_w_1 = np.asarray(w_1_for_w_0), np.asarray(w_0_for_w_1)

    ATE, ATE_unpaired_stderr = calculate_unpaired_treatment_effect(do_w_1, do_w_0)
    ATE_paired_stderr = np.std(np.asarray(ate_effects), ddof=1) / np.sqrt(
        len(ate_effects)
    )
    ATT, ATT_stderr = calculate_unpaired_treatment_effect(w_1, w_0_for_w_1)
    ATU, ATU_stderr = calculate_unpaired_treatment_effect(w_1_for_w_0, w_0)
    naive_effect, naive_effect_stderr = calculate_unpaired_treatment_effect(w_1, w_0)

    return {
        "ATE": ATE,
        "ATE_stderr": ATE_paired_stderr,
        "ATT": ATT,
        "ATT_stderr": ATT_stderr,
        "ATU": ATU,
        "ATU_stderr": ATU_stderr,
        "naive_effect": naive_effect,
        "naive_effect_stderr": naive_effect_stderr,
    }

def calculate_treatment_effects(
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
    original = effects_template.get("original", "rewritten rewrite")
    rewrite = effects_template.get("rewrite", "rewrite")

    Y1_count = 0
    Y1_rewards = []
    Y0_rewards = []
    for example in dataset.values():
        if example["w_original"]:
            Y1_count += 1
            Y1_rewards.append(example[effects_template["reward_key"]][original])
        else:
            Y0_rewards.append(example[effects_template["reward_key"]][rewrite])

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

    treatment_effects = calculate_average_treatment_effects(dataset, **effects_template)
    effects_template_naive = effects_template.copy()
    effects_template_naive["original"] = "original"
    treatment_effects_naive = calculate_average_treatment_effects(
        dataset, **effects_template_naive
    )

    for key, value in treatment_effects_naive.items():
        treatment_effects[f"{key}_naive"] = value

    pp = pprint.PrettyPrinter(indent=4)
    logging.info(f"Treatment effects: {pp.pformat(treatment_effects)}")
    logging.info(
        f"Calculated ATE (from ATT and ATU): {(Y1_count * treatment_effects['ATT'] + Y0_count * treatment_effects['ATU'])/len(dataset)}"
    )

    rewrite_effects = calculate_rewrite_effect(dataset, **effects_template)
    logging.info(f"Rewrite effects: {pp.pformat(rewrite_effects)}")

    effects = {**treatment_effects, **rewrite_effects}
    effects["reward_std"] = pooled_std
    effects["Y1_count"] = Y1_count
    effects["Y0_count"] = Y0_count
    return effects


if __name__ == "__main__":
    with open(SCRIPT_DIR / "config.yaml", "r") as f:
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
    dataset_path = SCORED_DIR / dataset_folder / dataset_filename if dataset_folder is not None else SCORED_DIR / dataset_filename
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset_from_json(dataset_path)

    logging.info("Dataset loaded. Calculating treatment effects...")
    effects = calculate_treatment_effects(dataset, **effects_template)

    effects.update(effects_template)

    results_filename = (
        EFFECTS_DIR / f"{EXPERIMENT_NAME}_treatment_effects_{FILE_ID}.json"
    )
    logging.info(f"Saving treatment effects to: {results_filename}")
    write_to_json(effects, results_filename)
