import numpy as np
from constants import logging
from datasets import load_dataset

eli5_path = "/net/projects/veitch/prompt_distributions/data/ELI5/all.jsonl"
eli5 = load_dataset("json", data_files=eli5_path, split="train")

logging.info(f"Loaded ELI5 dataset from: {eli5_path}")
logging.info(f"Number of examples in dataset: {len(eli5)}")

n_examples = 25000
logging.info(f"Using {n_examples} examples")

dataset = []
lengths = []
for example in eli5.select(range(n_examples)):
    question = example["title"]
    for answer in example["answers"]["text"]:
        lengths.append(len(answer))
        dataset.append(
            {
                "question": question,
                "text": answer,
            }
        )

median_length = int(
    np.median(np.asarray(lengths))
)  # Note: Convert to int to prevent serialization issues

logging.info(f"Median length: {median_length}")

dataset_template = {
    # "experiment_name": "eli5_length_armorm",
    "n_examples": n_examples,
    "original_completions": dataset,
    "w_strings": {
        "w=1": "longer",
        "w=0": "shorter",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: len(x["text"]) > median_length,
    "get_original_completion": lambda x: x["text"],
    "reward_question": lambda x: x["question"],
    "rewrite_prompt": """{original_completion}\n\n[Adjust this answer so it's {w_counterfactual_string}, but change *nothing* else. If the above answer is phrased as a question do not answer it — just rewrite the question following the same instructions.]""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
    # "reward_model_path": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    # "reward_model_name": "ArmoRM",
    # Note: This is a question-answering reward model so needs a prompt to score the completion
    # "get_reward": get_deberta_reward,
    # "_score_example": _score_example_armorm,
}
