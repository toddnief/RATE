"""Functions for calculating treatment effects from a scored dataset."""

import pprint
from pathlib import Path

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


def calculate_unpaired_treatment_effect(treated, untreated):
    treated_effect = np.mean(treated) - np.mean(untreated)
    stderr = np.sqrt(
        np.var(treated, ddof=1) / len(treated)
        + np.var(untreated, ddof=1) / len(untreated)
    )
    return treated_effect, stderr


def calculate_rewrite_effect(
    dataset, original="original", rewrite="rewritten rewrite", **effects_template
):
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


def calculate_average_treatment_effects(dataset, **effects_template):
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


def calculate_treatment_corrections(effects):
    assert "Y1_count" in effects, "Please use calculate_treatment_effects()"

    b0 = effects["rewrite_effect_w_0"]
    b0_stderr = effects["rewrite_effect_w_0_stderr"]
    b1 = effects["rewrite_effect_w_1"]
    b1_stderr = effects["rewrite_effect_w_1_stderr"]

    alpha = effects["Y1_count"] / (effects["Y1_count"] + effects["Y0_count"])

    bound_variance = lambda x, y: x**2 + y**2 + np.abs(2 * x * y)

    ATT_corrected = effects["ATT"] + b0
    ATT_corrected_stderr = np.sqrt(bound_variance(effects["ATT_stderr"], b0_stderr))

    ATU_corrected = effects["ATU"] - b1
    ATU_corrected_stderr = np.sqrt(bound_variance(effects["ATU_stderr"], -b1_stderr))

    ATE_corrected = alpha * ATT_corrected + (1 - alpha) * ATU_corrected
    ATE_corrected_alt = effects["ATE"] + alpha * b0 - b1 * (1 - alpha)
    assert np.allclose(
        ATE_corrected, ATE_corrected_alt
    ), "The two ATE calculations should be equal"
    ATE_stderr = effects["ATE_stderr"]
    ATE_corrected_stderr = np.sqrt(
        ATE_stderr**2
        + alpha**2 * b0_stderr**2
        + (1 - alpha) ** 2 * b1_stderr**2
        + alpha * np.abs(2 * ATE_stderr * b0_stderr)
        + (1 - alpha) * np.abs(2 * ATE_stderr * b1_stderr)
        + alpha * (1 - alpha) * np.abs(2 * b0_stderr * b1_stderr)
    )

    return {
        "ATT_corrected": ATT_corrected,
        "ATT_corrected_stderr": ATT_corrected_stderr,
        "ATU_corrected": ATU_corrected,
        "ATU_corrected_stderr": ATU_corrected_stderr,
        "ATE_corrected": ATE_corrected,
        "ATE_corrected_stderr": ATE_corrected_stderr,
    }


def calculate_adjustment_effects(dataset, **effects_template):
    "Perform adjustment for covariates. Condition on each covariate, calculate the treatment effects, and reweight."
    raise NotImplementedError("Adjustment effects calculation not implemented yet.")
    # Condition on the covariates

    # Run the treatment effects calculation for each subpopulation

    # Reweight the overall treatment effects by the proportion of each subpopulation

    # Compute the adjusted std_err

    # Return the results


def calculate_treatment_effects(dataset, adjustment=False, **effects_template):
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

    corrected_effects = calculate_treatment_corrections(effects)
    effects.update(corrected_effects)

    if adjustment:
        # Perform covariate adjustment
        adjusted_effects = calculate_adjustment_effects(dataset, **effects_template)
        effects.update(adjusted_effects)

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
    logging.info(f"Loading dataset from: {dataset_filename}")
    dataset = load_dataset_from_json(SCORED_DIR / "complete" / dataset_filename)

    logging.info("Dataset loaded. Calculating treatment effects...")
    effects = calculate_treatment_effects(dataset, **effects_template)

    effects.update(effects_template)

    results_filename = (
        EFFECTS_DIR / f"{EXPERIMENT_NAME}_treatment_effects_{FILE_ID}.json"
    )
    logging.info(f"Saving treatment effects to: {results_filename}")
    write_to_json(effects, results_filename)
