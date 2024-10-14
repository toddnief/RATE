from datasets import load_dataset

helpsteer = load_dataset("nvidia/HelpSteer")
original_completions = helpsteer["train"]

dataset_template = {
    "dataset_name": "helpsteer_complexity",  # Note: This is used in the filename
    "original_completions": original_completions,
    "n_examples": 25000,
    "w_strings": {
        "w=1": "more complex",
        "w=0": "less complex",
    },
    "w_classifier": lambda row: False
    if row["complexity"] in [0, 1, 2]
    else True
    if row["complexity"] in [3, 4]
    else None,
    "get_reward_question": lambda row: row["prompt"],
    "get_original_completion": lambda row: row["response"],
    "rewrite_prompt": """{original_completion}\n\nPlease adjust this response so it's {w_counterfactual_string}, but change *nothing* else. Only provide the response text, not an acknowledgement of the request, the prompt or any other context.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
