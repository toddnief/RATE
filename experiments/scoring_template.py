import importlib
import sys
from pathlib import Path

# TODO: Actually create a package — this is a hack to get the import working
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))


def load_scoring_utils(scoring_model):
    """Factory function to load scoring utilities based on scoring model in config"""
    module_name = f"scoring_templates.{scoring_model.lower()}"

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
