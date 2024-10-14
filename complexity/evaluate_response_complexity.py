from tqdm import tqdm
import json
import argparse
import os
import numpy as np
import random
import openai
import time
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

SYSTEM_PROMPT = """[System]
You are a helpful, harmless, and precise assistant for question answering. For each question please select the correct answer from the given options to the best of your knowledge."""


USER_PROMPT = """Please rate the language complexity of following content by choosing from one of the categories below:
[Query: ]
{query}
[Response A: simple, the response contains simple words and phrases that can be understood by any layperson]
[Response B: complex, the response contains complex words and phrases that require expert knowledge]
Please use your best judgement to make the selection. \
Rating: <"A" or "B">
"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_modelA", type=str)

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    return args


def clean(text, sep="###"):
    result = text.split(sep)[0]
    return result if len(result) > 0 else " "


def gpt4_eval(sys_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.7,
            #max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as ex:
        print(ex)
        time.sleep(3)
    return "error"


if __name__ == "__main__":
    args = get_args()

    data_directory = "/net/projects/veitch/prompt_distributions/data"
    path = os.path.join(data_directory, f"{args.run_modelA}.json")
    output_path = os.path.join(data_directory, f"Complexity_Evals_GPT4_{args.run_modelA}.json")
    generations_A = json.load(open(path, "r"))

    #selected_indices = random.sample(range(len(generations_modelA)), 5)
    #generations_A = [generations_modelA[i] for i in selected_indices]

    for gen_A in tqdm(generations_A, total=len(generations_A)):
        prompt = gen_A["question"]
        response_modelA = gen_A["baseline_query_completion"]
        print ("Response:", response_modelA)
        content = gpt4_eval(sys_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT.format(query=response_modelA))
        print("Content", content)
        try:
            rating = content.split(":")[1].strip()
            print("Rating:", rating)
            if rating == "A":
                llm_judgement = "simple"
            elif rating == "B":
                llm_judgement = "complex"
        except Exception as e:
            print("could not parse answer")
            llm_judgement = None
        
        eval_row = gen_A.copy()
        eval_row.update({
            "llm_answer": content,
            "rating": rating,
            "llm_judgement": llm_judgement,
            })
        
        with open(output_path, "a") as f:
            f.write(json.dumps(eval_row) + "\n")
    