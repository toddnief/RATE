import os
import json
import time
from tqdm import tqdm
import argparse

# even though transformers isn't used directly, it's necessary (?!) for some reason to avoid 
# ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
import transformers 

from datasets import load_dataset

from openai import OpenAI
import tiktoken

import random

from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

def get_score(model, question, answer):
    inputs = tokenizer(question, answer, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    return logits[0][0].item()


############################################ Settings ############################################

parser = argparse.ArgumentParser(description="Generate baseline, conditional, and interventional completions for movie reviews using GPT-4 model.")

# Add the arguments
parser.add_argument('--model_id', type=str, default="gpt-4-1106-preview", help='Model ID for the GPT-4 model.')
parser.add_argument('--subset_size', type=int, default=300, help='Number of samples for the dataset.')
parser.add_argument('--seed', type=int, default=123, help='Random seed for sampling the dataset.')
parser.add_argument('--num_tokens', type=int, default=40, help='Number of tokens to use from each review.')
parser.add_argument('--num_next_tokens', type=int, default=1, help='Number of tokens to use for the next text after the review.')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling completions.')
parser.add_argument('--job_id', type=int, default=0, help='Job ID for the current run.')
args = parser.parse_args()

model_id = args.model_id
subset_size = args.subset_size
seed = args.seed
num_tokens = args.num_tokens
num_next_tokens = args.num_next_tokens
temperature = args.temperature
job_id = args.job_id

# filename to save the prompts and completions
data_directory = "/net/projects/veitch/prompt_distributions/data"
save_file = f"prompt_complexity_completions_{model_id}_subset{subset_size}_seed{seed}_numtokens{num_tokens}_next{num_next_tokens}_temp{temperature}_job{job_id}.json"

############################################ Check Data Directory ############################################

#assert os.path.exists(data_directory), f"Folder '{data_directory}' does not exist."
#full_save_path = os.path.join(data_directory, save_file)

# Double check that the file does not already exist
#assert not os.path.exists(full_save_path), f"File '{full_save_path}' already exists."

#print("Data will be saved to:", full_save_path)


############################################ GPT-4 Client ############################################
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def gpt4_completion(user_prompt: str) -> str:
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
            #max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as ex:
        print(ex)
        time.sleep(3)
    return "error"

############################################ Prompt Templates ############################################

baseline_template = lambda question: f"""
{question}
[Please answer this question.]
Answer: """

conditional_template = lambda question, age_level: f""" 
{question}
[Please answer this question like I am {age_level}.]
Answer: """

interventional_template = lambda flipped_label, baseline_completion: f"""
Original Answer: 
{baseline_completion}
[Adjust the original answer so it contains {flipped_label}, but change *nothing* else.]
Adjusted Answer:"""

############################################ ELI5 Dataset ############################################

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

data_directory = "/net/projects/veitch/prompt_distributions/data"
data_path = os.path.join(data_directory, "Complexity_Evals_GPT4_prompt_complexity_completions_gpt-4-1106-preview_subset300_seed123_numtokens40_next1_temp0.7_job0.json")
data = read_jsonlist(data_path)
print("Example question: ", data[0])
print("Example question: ", data[1])

output_path = os.path.join(data_directory, f"Complexity_Evals_GPT4_prompt_interventional_completions_ranked.json")

# Get the tokenizer for GPT-4
#tokenizer = tiktoken.encoding_for_model("gpt-4")

############################################ Generate Prompts, Completions ############################################

prompts = []
# For each review, generate the baseline, conditional, and interventional queries
# Start with the positive reviews
print("Generating answers...")
for item in tqdm(data):
    prompt = item["question"]
    baseline_completion = item["baseline_query_completion"]
    llm_rating = item["rating"]
    llm_judgement = item["llm_judgement"]

    if llm_judgement == "complex":
        flipped_llm_judgement = "simple"
        interventional_label = "simple words and phrases that can be understood by any layperson"
    elif llm_judgement == "simple":
        flipped_llm_judgement = "complex"
        interventional_label = "complex words and phrases that require expert knowledge"

    interventional_query = interventional_template(interventional_label, baseline_completion)
    interventional_query_completion = gpt4_completion(interventional_query)

    eval_row = item.copy()
    eval_row.update({
        "interventional_query": interventional_query,
        "interventional_query_completion": interventional_query_completion,
        "interventional_query_completion_reward_score": get_score(rank_model, prompt, interventional_query_completion),
        "baseline_query_completion_reward_score": get_score(rank_model, prompt, baseline_completion),
        })
        
    with open(output_path, "a") as f:
        f.write(json.dumps(eval_row) + "\n")

    