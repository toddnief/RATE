from constants import HH_RLHF_PATH
from datasets import load_dataset

original_completions = load_dataset("json", data_files=str(HH_RLHF_PATH))
# Note: The first half of the dataset is the "harmfulness" data
original_completions = original_completions["train"]["rejected"][:25000]


dataset_template = {
    "dataset_name": "hh_rlhf_harmlessness",  # Note: This is used in the filename
    "original_completions": original_completions,
    "n_examples": 10,
    "w_strings": {
        "w=1": "more harmful",
        "w=0": "less harmful",
    },
    "w_classifier": lambda x: True,  # Calculating ATT only so always return True
    "get_reward_question": lambda row: row.split("Human: ")[1]
    .split("Assistant: ")[0]
    .strip(),
    "get_original_completion": lambda row: row.split("Assistant: ")[1]
    .rsplit("Human:", 1)[
        0
    ]  # Note: Some completions have more than one question and response
    .strip(),
    "rewrite_prompt": """{original_completion}\n\n[Adjust this response so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
