from constants import HH_RLHF_PATH
from datasets import load_dataset

hh_rlhf = load_dataset("json", data_files=str(HH_RLHF_PATH))
# Note: The back half of the dataset is the "helpfulness" data
# Also note: We are getting two completions per example
# The "chosen" are classified as helpful, the "rejected" are classified as unhelpful
hh_rlhf = hh_rlhf["train"][-12500:]

original_completions = []
for key, completions in hh_rlhf.items():
    w_original = key == "chosen"
    for completion in completions:
        split_completion = completion.rsplit("Assistant: ", 1)
        reward_question = split_completion[0].strip()
        original_completion = split_completion[
            1
        ].strip()  # Note: last assistant response
        original_completions.append(
            {
                "w_original": w_original,
                "reward_question": reward_question + " Assistant: ",
                "original_completion": original_completion,
            }
        )


dataset_template = {
    "dataset_name": "hh_rlhf_helpfulness",  # Note: This is used in the filename
    "original_completions": original_completions,
    "n_examples": 10,
    "w_strings": {
        "w=1": "more helpful",
        "w=0": "less helpful",
    },
    "w_classifier": lambda row: row["w_original"],
    "get_reward_question": lambda row: row["reward_question"],
    "get_original_completion": lambda row: row["original_completion"],
    "rewrite_prompt": """{original_completion}\n\nAdjust this response so it's {w_counterfactual_string} than the current response, but change *nothing* else. By "helpful", I mean informative, relevant and clear.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
