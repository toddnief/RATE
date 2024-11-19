"""Scoring template factory function to load scoring utilities based on scoring model in config"""

import importlib
from typing import Any, Dict

# TODO: Actually create a package — this is a hack to get the import working
# SCRIPT_DIR = Path(__file__).resolve().parent
# sys.path.append(str(SCRIPT_DIR))


def load_scoring_utils(scoring_model: str) -> Dict[str, Any]:
    """
    Factory function to load scoring utilities based on the provided scoring model

    Args:
        scoring_model: The name of the scoring model to load (as a string). It will be converted to lowercase
                       and used to import the module from the `scoring_templates` package.

    Returns:
        A dictionary containing:
            - "reward_model": The scoring model's reward model object.
            - "reward_tokenizer": The tokenizer used by the reward model.
            - "reward_model_path": The path to the reward model.
            - "reward_model_name": The name of the reward model.
            - "_score_example": A function to score an example using the reward model.

    Raises:
        ValueError: If the specified scoring model cannot be found or imported.
    """
    module_name = f"rate.scoring_templates.{scoring_model.lower()}"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Scoring model {scoring_model} not found")

    return {
        "reward_model": module.reward_model,
        "reward_tokenizer": module.reward_tokenizer,
        "reward_model_path": module.reward_model_path,
        "reward_model_name": module.reward_model_path,
        "_score_example": module._score_example,
    }
