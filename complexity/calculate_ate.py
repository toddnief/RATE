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


# Load the dataset
def read_jsonlist(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    data = []
    with open(filename, "r") as fin:
        for line in fin:
            #print(line)
            data.append(json.loads(line))
    print(len(data))
    return data

def calculate_ate(dataset):
    
    do_w_1 = 0
    do_w_0 = 0

    w_1 = 0
    w_0 = 0
    
    w_1_count = 0

    for item in tqdm(dataset):

        prompt = item["question"]
        baseline_completion = item["baseline_query_completion"]
        baseline_reward_score = item["baseline_query_completion_reward_score"]
        interventional_completion = item["interventional_query_completion"]
        interventional_reward_score = item["interventional_query_completion_reward_score"]
        llm_judgement = item["llm_judgement"]
        llm_rating = item["rating"]

        if llm_rating == "A" and llm_judgement == "simple":
            do_w_1 += baseline_reward_score
            do_w_0 += interventional_reward_score
            w_1_count += 1

            w_1 += baseline_reward_score
        elif llm_rating == "B" and llm_judgement == "complex":
            do_w_1 += interventional_reward_score
            do_w_0 += baseline_reward_score

            w_0 += baseline_reward_score

    n = len(dataset)
    do_w_1_mean = do_w_1 / n
    do_w_0_mean = do_w_0 / n

    ATE = do_w_1_mean - do_w_0_mean

    w_1_mean = w_1 / w_1_count
    w_0_mean = w_0 / (n - w_1_count)
    correlational_effect = w_1_mean - w_0_mean

    do_w_1_ssd = 0  # Sum of squared deviations for do_w_1
    do_w_0_ssd = 0  # Sum of squared deviations for do_w_0

    for example in dataset:

        if example["llm_judgement"] == "simple":
            do_w_1_diff = float(example["baseline_query_completion_reward_score"]) - do_w_1_mean
            do_w_0_diff = float(example["interventional_query_completion_reward_score"]) - do_w_0_mean
        else:
            do_w_1_diff = float(example["interventional_query_completion_reward_score"]) - do_w_1_mean
            do_w_0_diff = float(example["baseline_query_completion_reward_score"]) - do_w_0_mean

        do_w_1_ssd += do_w_1_diff**2
        do_w_0_ssd += do_w_0_diff**2

    do_w_1_var = do_w_1_ssd / (n - 1)
    do_w_0_var = do_w_0_ssd / (n - 1)

    return ATE, do_w_1_var, do_w_0_var, correlational_effect


if __name__ == "__main__":
    args = get_args()

    data_directory = "/net/projects/veitch/prompt_distributions/data"
    file_path = os.path.join(data_directory, f"Complexity_Evals_GPT4_prompt_interventional_completions_ranked.json")
    data = read_jsonlist(file_path)


    ATE, do_w_1_var, do_w_0_var, correlational_effect = calculate_ate(data)

    print("ATE:", ATE)
    print("Variance of do_w_1:", do_w_1_var)
    print("Variance of do_w_0:", do_w_0_var)
    print("Correlational effect:", correlational_effect)

    """
    do_w_1 = 0
    do_w_0 = 0
    
    for item in tqdm(data):
        prompt = item["question"]
        baseline_completion = item["baseline_query_completion"]
        baseline_reward_score = item["baseline_query_completion_reward_score"]
        interventional_completion = item["interventional_query_completion"]
        interventional_reward_score = item["interventional_query_completion_reward_score"]
        llm_judgement = item["llm_judgement"]
        llm_rating = item["rating"]

        if llm_rating == "A" and llm_judgement == "simple":
            do_w_1 += baseline_reward_score
            do_w_0 += interventional_reward_score
        elif llm_rating == "B" and llm_judgement == "complex":
            do_w_1 += interventional_reward_score
            do_w_0 += baseline_reward_score

    do_w_1_mean = do_w_1 / len(data)
    do_w_0_mean = do_w_0 / len(data)

    diff = (do_w_1 - do_w_0) / len(data)
    w1 = do_w_1 / len(data)
    w2 = do_w_0 / len(data)
    print("diff", diff, "w1", w1, "w2", w2)

    do_w_1_ssd = 0  # Sum of squared deviations for do_w_1
    do_w_0_ssd = 0  # Sum of squared deviations for do_w_0
    covariance = 0

    for example in tqdm(data):
        if example["llm_judgement"] == "simple":
            do_w_1_diff = float(example["baseline_query_completion_reward_score"]) - do_w_1_mean
            do_w_0_diff = float(example["interventional_query_completion_reward_score"]) - do_w_0_mean
        else:
            do_w_1_diff = float(example["interventional_query_completion_reward_score"]) - do_w_1_mean
            do_w_0_diff = float(example["baseline_query_completion_reward_score"]) - do_w_0_mean

        do_w_1_ssd += do_w_1_diff ** 2
        do_w_0_ssd += do_w_0_diff ** 2
        covariance += do_w_1_diff * do_w_0_diff

    do_w_1_var = do_w_1_ssd / (len(data) - 1)
    do_w_0_var = do_w_0_ssd / (len(data) - 1)
    covariance /= (len(data) - 1)

    print(do_w_1_var, do_w_0_var, covariance)
    """
