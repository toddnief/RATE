import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rate.constants import DEVICE

reward_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, trust_remote_code=True
).to(DEVICE)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)


def _score_example(
    model,
    tokenizer,
    question,
    answer,
    device=DEVICE,
    truncation=True,
):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    with torch.no_grad():
        with autocast():
            model = model.to(device)
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            ).to(device)
            outputs = model(inputs)
            reward = outputs.score.float().item()
    del inputs, outputs  # Explicitly free up memory to prevent OOM
    return reward
