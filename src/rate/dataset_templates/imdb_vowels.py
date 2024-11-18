"""Template for synthetic experiment introducing correlation between typos and on-target attribute of starting with a vowel."""

import random

from datasets import load_dataset

dataset = load_dataset("imdb")


def starts_with_vowel(text):
    return text[0] in set("aeiouAEIOU") if text else False


def add_typos(text, min_typos=0, max_typos=7):
    """
    Introduces typos in the input text by swapping adjacent letters.
    The number of typos is chosen uniformly at random within the given range.

    Args:
        text (str): The input text where typos will be added.
        min_typos (int): Minimum number of typos to introduce.
        max_typos (int): Maximum number of typos to introduce.

    Returns:
        str: The modified text with typos.
    """
    # Choose a random number of typos within the range
    num_typos = random.randint(min_typos, max_typos)

    text_as_list = list(text)  # Convert the text to a mutable list
    for _ in range(num_typos):
        # Find a random position to introduce a typo
        typo_index = random.randint(
            0, len(text_as_list) - 2
        )  # Ensure room to swap with the next character
        if (
            text_as_list[typo_index].isalpha()
            and text_as_list[typo_index + 1].isalpha()
        ):
            # Swap the current character with the next one
            text_as_list[typo_index], text_as_list[typo_index + 1] = (
                text_as_list[typo_index + 1],
                text_as_list[typo_index],
            )
    return "".join(text_as_list)


# Introduce typos to examples that start with a vowel
def process_example(example):
    if starts_with_vowel(example["text"]):
        example["text"] = add_typos(example["text"])
    return example


dataset = dataset.map(process_example, desc="Adding typos to dataset")

dataset_template = {
    "dataset_name": "imdb_vowels",
    "n_examples": 500,  # Note: The number of examples to rewrite
    "original_completions": dataset["train"],
    "w_strings": {
        "w=1": "starts with a vowel",
        "w=0": "doesn't start with a vowel",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": lambda x: starts_with_vowel(x["text"]),
    "get_original_completion": lambda x: x["text"],
    "get_reward_question": lambda x: "Write a movie review: ",  # Note: The reward models need a prompt to score completions
    "rewrite_prompt": """{original_completion}\n\nAdjust this review so it {w_counterfactual_string}, but change *nothing* else other than minor changes to make the first sentence grammatically correct.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
}
