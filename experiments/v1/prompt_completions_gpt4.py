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
parser.add_argument('--generate_locally', action='store_true', help='Generate completions locally instead of using OpenAI API.')
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
save_file = f"prompt_completions_{model_id}_subset{subset_size}_seed{seed}_numtokens{num_tokens}_next{num_next_tokens}_temp{temperature}_job{job_id}.json"

############################################ Check Data Directory ############################################

assert os.path.exists(data_directory), f"Folder '{data_directory}' does not exist."
full_save_path = os.path.join(data_directory, save_file)

# Double check that the file does not already exist
assert not os.path.exists(full_save_path), f"File '{full_save_path}' already exists."

print("Data will be saved to:", full_save_path)


############################################ GPT-4 Client ############################################

import dotenv
dotenv.load_dotenv()

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

baseline_template = lambda review, next_text: f"""Movie Review: 
{review}
[Finish this movie review.]
Completion: {next_text}"""

conditional_template = lambda review, flipped_sentiment, next_text: f"""Movie Review: 
{review}
[Finish this movie review with {flipped_sentiment} sentiment.]
Completion: {next_text}"""

interventional_template = lambda review, flipped_sentiment, next_text, baseline_completion: f"""Movie Review: 
{review}
[Finish the movie review.]
Original Completion: 
{next_text+baseline_completion}
[Adjust the original completion so the sentiment is {flipped_sentiment}, but change *nothing* else.]
Adjusted Completion: {next_text}"""

############################################ IMDB Dataset ############################################

# Load the dataset and sample a subset for both train and test splits
imdb = load_dataset('imdb')

# Randomly sample a subset of the data. For now, just use the train split.
train_subset = imdb['train'].shuffle(seed=seed).select(range(subset_size))

# Get the tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Truncate the reviews and split them into positive and negative
# (Splitting the reviews by sentiment is not necessary, but it makes it easier to make sure we have the right proportion of each sentiment.)
# (Also, splitting up the sentiment makes it easier to inspect the .json file)
pos_reviews, neg_reviews = utils.process_reviews(train_subset, tokenizer, num_tokens=num_tokens, num_next_tokens=num_next_tokens)

print("Number of positive in the subset:", len(pos_reviews))
print("Number of negative in the subset:", len(neg_reviews))

print("Example positive review:\n", pos_reviews[0][0])
print("Example negative review:\n", neg_reviews[0][0])

############################################ Load Model for Local Generation ############################################

# TODO: Just using Gemma for now — could set up multiple options here in the future

if args.generate_locally:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

############################################ Generate Prompts, Completions ############################################

prompts = []
# For each review, generate the baseline, conditional, and interventional queries
# Start with the positive reviews
print("Generating completions for positive reviews...")
for (review, next_text) in tqdm(pos_reviews):
    label = 1
    sentiment = "positive"
    flipped_sentiment = "negative"

    baseline_query = baseline_template(review, next_text)
    if args.generate_locally:
        input_ids = tokenizer(baseline_query, return_tensors="pt").to(model.device)
        outputs = model.generate(max_length=400, **input_ids)
        baseline_query_completion = tokenizer.decode(outputs[0])
    else:
        baseline_query_completion = gpt4_completion(baseline_query)

    conditional_query = conditional_template(review, flipped_sentiment, next_text)
    conditional_query_completion = gpt4_completion(conditional_query)

    interventional_query = interventional_template(review, flipped_sentiment, next_text, baseline_query_completion)
    interventional_query_completion = gpt4_completion(interventional_query)

    prompts.append({
            "review": review,
            "next_text": next_text,
            "label": label,
            "sentiment": sentiment,
            "flipped_sentiment": flipped_sentiment,
            "baseline_query": baseline_query,
            "baseline_query_completion": baseline_query_completion,
            "conditional_query": conditional_query,
            "conditional_query_completion": conditional_query_completion,
            "interventional_query": interventional_query,
            "interventional_query_completion": interventional_query_completion,
        })

# Now do the same for the negative reviews
print("Generating completions for negative reviews...")
for (review, next_text) in tqdm(neg_reviews):
    label = 0
    sentiment = "negative"
    flipped_sentiment = "positive"

    baseline_query = baseline_template(review, next_text)
    baseline_query_completion = gpt4_completion(baseline_query)

    conditional_query = conditional_template(review, flipped_sentiment, next_text)
    conditional_query_completion = gpt4_completion(conditional_query)

    interventional_query = interventional_template(review, flipped_sentiment, next_text, baseline_query_completion)
    interventional_query_completion = gpt4_completion(interventional_query)

    prompts.append({
            "review": review,
            "next_text": next_text,
            "label": label,
            "sentiment": sentiment,
            "flipped_sentiment": flipped_sentiment,
            "baseline_query": baseline_query,
            "baseline_query_completion": baseline_query_completion,
            "conditional_query": conditional_query,
            "conditional_query_completion": conditional_query_completion,
            "interventional_query": interventional_query,
            "interventional_query_completion": interventional_query_completion,
        })

############################################ Save Prompts and Completions ############################################

# Save the prompts and completions to a JSON file
json.dump(prompts, open(full_save_path, "w"), indent=2)

# Avoid accidental overwrites by setting the file to read-only
os.chmod(full_save_path, 0o444)