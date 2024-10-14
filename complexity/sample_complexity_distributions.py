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

from utils import utils
import random

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

assert os.path.exists(data_directory), f"Folder '{data_directory}' does not exist."
full_save_path = os.path.join(data_directory, save_file)

# Double check that the file does not already exist
assert not os.path.exists(full_save_path), f"File '{full_save_path}' already exists."

print("Data will be saved to:", full_save_path)


############################################ GPT-4 Client ############################################
os.environ["OPENAI_API_KEY"] = "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"
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

#conditional_template = lambda question, age_level: f""" 
#{question}
#[Please answer this question like I am {age_level}.]
#Answer: """

#interventional_template = lambda question, flipped_age_level, baseline_completion: f"""
#{question}
#[Please answer this question.]
#Original Answer: 
#{baseline_completion}
#[Adjust the original answer so it can be understoood by a {flipped_age_level}, but change *nothing* else.]
#Adjusted Answer:"""

############################################ IMDB Dataset ############################################

# Load the dataset
def read_jsonlist(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    data = []
    with open(filename, "r") as fin:
        for line in fin:
            #print(line)
            data.append(json.loads(line)["title"])
    print(len(data))
    return data

data = read_jsonlist("/net/projects/veitch/garbacea/ELI5/all.jsonl")
data = data[:200]
print("Example question: ", data[0])
print("Example question: ", data[1])

# Get the tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")

############################################ Generate Prompts, Completions ############################################

prompts = []
# For each review, generate the baseline, conditional, and interventional queries
# Start with the positive reviews
print("Generating answers...")
for question in tqdm(data):
    #age_level = "an academic"
    #flipped_age_level = "a 5 year old"

    baseline_query = baseline_template(question)
    baseline_query_completion = gpt4_completion(baseline_query)

    #conditional_query = conditional_template(question, age_level)
    #conditional_query_completion = gpt4_completion(conditional_query)

    #interventional_query = interventional_template(question, flipped_age_level, baseline_query_completion)
    #interventional_query_completion = gpt4_completion(interventional_query)

    prompts.append({
            "question": question,
            #"age_level": age_level,
            #"flipped_age_level": flipped_age_level,
            "baseline_query": baseline_query,
            "baseline_query_completion": baseline_query_completion,
            #"conditional_query": conditional_query,
            #"conditional_query_completion": conditional_query_completion,
            #"interventional_query": interventional_query,
            #"interventional_query_completion": interventional_query_completion,
        })

############################################ Save Prompts and Completions ############################################

# Save the prompts and completions to a JSON file
json.dump(prompts, open(full_save_path, "w"), indent=2)

# Avoid accidental overwrites by setting the file to read-only
os.chmod(full_save_path, 0o444)