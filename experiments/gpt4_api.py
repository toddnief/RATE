import sys
import time

import numpy as np
from constants import logging


def get_gpt4_completion(
    client, user_prompt: str, temperature=0.5, model_id="gpt-4o-mini", max_tokens=2048
) -> str:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as ex:
        logging.info(ex)
    return "error"


def get_rewritten_completion(
    original_completion, rewrite_prompt, w_counterfactual_string, client
):
    return get_gpt4_completion(
        client,
        rewrite_prompt.format(
            original_completion=original_completion,
            w_counterfactual_string=w_counterfactual_string,
        ),
    )


def create_api_batch(dataset, completion_key, rewritten_rewrite=False, **kwargs):
    w_strings = kwargs.get("w_strings")
    rewrite_prompt = kwargs.get("rewrite_prompt")

    batch_input = []
    for id, example in dataset.items():
        completion_to_rewrite = example["completions"][completion_key]
        # Note: flip w_original for rewrite of rewrite
        w_original = (
            example["w_original"]
            if not rewritten_rewrite
            else not example["w_original"]
        )
        # TODO: How should we handle None values? (ie if our W classifier fails an assertion)
        w_counterfactual_string = w_strings["w=0" if w_original else "w=1"]
        batch_input.append(
            {
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": kwargs.get("model"),
                    "messages": [
                        {
                            "role": "user",
                            "content": rewrite_prompt.format(
                                original_completion=completion_to_rewrite,
                                w_counterfactual_string=w_counterfactual_string,
                            ),
                        }
                    ],
                    "temperature": kwargs.get("temperature"),
                },
            }
        )

    return batch_input


def submit_batch(client, batch_input_filename, file_id, n_examples, **kwargs):
    batch_input_file = client.files.create(
        file=open(batch_input_filename, "rb"), purpose="batch"
    )
    logging.info("Submitting batch to OpenAI API. Waiting...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "test run"},
    )

    time.sleep(15)
    start = time.time()
    status = client.batches.retrieve(batch.id).status
    while status != "completed":
        time.sleep(10)
        status = client.batches.retrieve(batch.id).status
        if status == "failed":
            logging.info("Batch failed")
            logging.info(client.batches.retrieve(batch.id).errors)
            sys.exit(1)
        logging.info(
            f"Status: {status}. Time elapsed: {np.round((time.time() - start)/60,1)} minutes. Completed requests: {client.batches.retrieve(batch.id).request_counts.completed}"
        )

    return client.files.content(client.batches.retrieve(batch.id).output_file_id)
