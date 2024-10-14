from datasets import load_dataset

dataset = load_dataset("imdb")

dataset_template = {
    "dataset_name": "imdb_sentiment",
    "n_examples": 25000,
    "original_completions": dataset["train"],
    "w_strings": {
        "w=1": "positive sentiment",
        "w=0": "negative sentiment",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: x["label"] == 1,
    "get_original_completion": lambda x: x["text"],
    # "reward_model_path": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    # "reward_model_name": "ArmoRM",
    "reward_question": lambda x: "Write a movie review: ",  # Note: This is a question-answering reward model so needs a prompt to score the completion
    # "get_reward": get_armorm_reward,
    # "_score_example": _score_example_armorm,
    "rewrite_prompt": """{original_completion}\n\n[Adjust this review so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
