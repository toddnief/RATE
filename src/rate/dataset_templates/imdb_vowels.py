"""Template for synthetic experiment introducing correlation between typos and on-target attribute of starting with a vowel."""

import random

from datasets import load_dataset

dataset = load_dataset("imdb")


def starts_with_vowel(text):
    return text[0] in set("aeiouAEIOU") if text else False

TYPO_PERCENTAGE = 30
def add_typos(text, typo_percentage=TYPO_PERCENTAGE):
    """
    Introduces typos in a percentage of words by swapping adjacent letters.
    The percentage of words affected is given by typo_percentage.

    Args:
        text (str): The input text where typos will be added.
        typo_percentage (int): Percentage of words to introduce typos to (default 10%).

    Returns:
        str: The modified text with typos.
    """
    words = text.split()  # Split the text into words
    num_words = len(words)
    num_typos = int(num_words * typo_percentage / 100)  # Calculate number of words to affect
    if num_typos == 0:
        return text
    print(f"Adding typos to {num_typos} words")

    # Randomly select words to introduce typos
    words_to_typo = random.sample(range(num_words), num_typos)

    for idx in words_to_typo:
        word = words[idx]
        if len(word) > 2:  # Only add typos to words with more than two characters
            typo_index = random.randint(0, len(word) - 2)  # Select random index to swap
            if word[typo_index].isalpha() and word[typo_index + 1].isalpha():
                # Swap adjacent characters
                word_as_list = list(word)
                word_as_list[typo_index], word_as_list[typo_index + 1] = (
                    word_as_list[typo_index + 1],
                    word_as_list[typo_index],
                )
                words[idx] = "".join(word_as_list)

    return " ".join(words)


# Introduce typos to examples that start with a vowel
def process_example(example):
    if starts_with_vowel(example["text"]):
        example["text"] = add_typos(example["text"])
    return example


dataset = dataset.map(process_example, desc="Adding typos to dataset")

dataset_template = {
    "dataset_name": f"imdb_vowels_{TYPO_PERCENTAGE}typos",
    "n_examples": 2500,  # Note: The number of examples to rewrite
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
