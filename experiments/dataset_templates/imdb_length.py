"""Template for rewriting the IMDB dataset on target attribute length."""

from datasets import load_dataset

dataset = load_dataset("imdb")

dataset_template = {
    "dataset_name": "imdb_length",
    "original_completions": dataset["train"],
    "w_strings": {
        "w=1": "longer",
        "w=0": "shorter",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: len(x) > 970,
    "get_original_completion": lambda x: x["text"],
    "reward_question": lambda x: "Write a movie review: ",  # Note: The reward models need a prompt to score completions
    "rewrite_prompt": """{original_completion}\n\n[Adjust this review so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
