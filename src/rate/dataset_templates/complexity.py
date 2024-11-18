import torch
from deberta_utils import get_deberta_reward
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llama3_tokenizer = AutoTokenizer.from_pretrained(model_id)
llama3 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

system_prompt = {
    "role": "system",
    "content": "You are a helpful, harmless, and precise assistant for question answering. For each question please select the correct answer from the given options to the best of your knowledge. Do not add additional context or information. Only respond with the choice that corresponds to the correct option.",
}

user_prompt = {
    "role": "user",
    "content": """
        Please rate the language complexity of following content by choosing from one of the categories below:
        [Query: ]
        {text}
        [Response A: simple, the response contains simple words and phrases that can be understood by any layperson]
        [Response B: complex, the response contains complex words and phrases that require expert knowledge]
        Please use your best judgement to make the selection. \
        Rating: <"A" or "B">
        """,
}


def llama_complexity(
    text,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    model=llama3,
    tokenizer=llama3_tokenizer,
    positive_label="A",
    negative_label="B",
):
    user_prompt["content"] = user_prompt["content"].format(text=text)
    messages = [
        system_prompt,
        user_prompt,
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)

    if decoded_response == positive_label:
        return True
    elif decoded_response == negative_label:
        return False
    # TODO: Sort this out
    else:
        return False


dataset_template = {
    "experiment_name": "llama_complexity_smoke_test",
    "w_strings": {
        "w=1": "complex",
        "w=0": "simple",
    },  # Note: if w=1 for the original completion, rewrite using w=0
    "w_classifier": llama_complexity,
    "get_original_completion": lambda x: x["text"],
    "reward_model_path": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "reward_question": "Write a movie review: ",  # Note: Some models are question-answering reward models so need a prompt to score the completion
    "get_reward": get_deberta_reward,
    "rewrite_prompt": """{original_completion}\n\n[Adjust this review so it's {w_counterfactual_string}, but change *nothing* else.""",
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.7,
    "max_batch_size": 25,
}
