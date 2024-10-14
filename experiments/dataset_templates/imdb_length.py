from datasets import load_dataset

# from utils_armorm import ARMORM_PATH, _score_example_armorm  # noqa


# Note: 0 is negative sentiment, 1 is positive sentiment
# def get_distilbert_logits(outputs, idx=0):
#     return outputs.logits[0][idx].item()


dataset = load_dataset("imdb")

dataset_template = {
    "dataset_name": "imdb_length_armorm",
    "original_completions": dataset["train"],
    "w_strings": {
        "w=1": "longer",
        "w=0": "shorter",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: len(x) > 970,
    "get_original_completion": lambda x: x["text"],
    "rewrite_prompt": """{original_completion}\n\n[Adjust this review so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
    # "reward_model_path": ARMORM_PATH,
    # "reward_model_name": "ArmoRM",
    # "reward_question": "Write a movie review: ",  # Note: This is a question-answering reward model so needs a prompt to score the completion
    # "get_reward": get_distilbert_logits,
    # "_score_example": _score_example_armorm,
}
