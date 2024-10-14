import torch
from constants import HH_RLHF_PATH
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

original_completions = load_dataset("json", data_files=str(HH_RLHF_PATH))
# Note: The back half of the dataset is the "helpfulness" data
original_completions = original_completions["train"]["rejected"][-25000:]


tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def classify_sentiment(completion, model=model, tokenizer=tokenizer):
    inputs = tokenizer(completion, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax().item() == 1


dataset_template = {
    "dataset_name": "hh_rlhf_sentiment",
    "original_completions": original_completions,
    "n_examples": 25000,
    "w_strings": {
        "w=1": "positive sentiment",
        "w=0": "negative sentiment",
    },
    "w_classifier": classify_sentiment,
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
