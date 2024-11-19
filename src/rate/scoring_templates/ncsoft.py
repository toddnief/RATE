import torch
from constants import DEVICE
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, pipeline

reward_model_path = "NCSOFT/Llama-3-OffsetBias-RM-8B"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
reward_model = pipeline(
    "sentiment-analysis",  # Note: This is a reward model but they use this in their demo
    model=reward_model_path,
    device=DEVICE,
    tokenizer=reward_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)


# TODO: Set something up in a utils file that generalizes this
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
            pipe_kwargs = {
                "top_k": None,
                "function_to_apply": "none",
                "batch_size": 1,
            }

            inputs = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                ).replace(tokenizer.bos_token, "")
            ]
            outputs = model(inputs, **pipe_kwargs)
            rewards = [output[0]["score"] for output in outputs]
            reward = rewards[0]
    del inputs, outputs  # Explicitly free up memory to prevent OOM
    return reward
