# Load imdb for examples
from datasets import load_dataset
import pandas as pd
import numpy as np

import sys

# Batch API related imports
from openai import OpenAI # OpenAI API
import json # for creating batch files

# And for loading the API key from the .env file
from dotenv import load_dotenv
import os
import time

sample_size = int(sys.argv[1])
# sample_size = 100 for testing, 50000 for full dataset

######################## Load IMDB dataset ######################

# load imdb dataset from huggingface
dataset = load_dataset('imdb')

# Creating separate DataFrames for train and test data
train_df = pd.DataFrame({'text': dataset['train']['text'], 'label': dataset['train']['label']})
test_df = pd.DataFrame({'text': dataset['test']['text'], 'label': dataset['test']['label']})

# Concatenate the DataFrames
imdb = pd.concat([train_df, test_df], ignore_index=True)

# # Remove samples which have length greater than 4000 characters
# imdb = imdb[imdb['text'].apply(lambda x: len(x) < 4000)]

# # Reset the index after filtering
# imdb = imdb.reset_index(drop=True)

# Bin text lengths
imdb.loc[:, 'text_length'] = imdb['text'].str.len()
# set the length threshold to be the median length
length_threshold = imdb['text_length'].median()
print(f"Median length of text: {length_threshold}")
imdb['is_long'] = imdb['text_length'] > length_threshold

# make sure is_long got assigned correctly
print(imdb['is_long'].value_counts())
assert (imdb['is_long'] == (imdb['text'].str.len() > length_threshold)).all()

###################### Prepare the Batch ######################

# Get rewrites from OpenAI batch API for entire imdb dataset, while preserving the order (index) of the dataset

# Setup the OpenAI client
load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def review_messages(review, is_long):
    """ Prompt to the model to rewrite the review to be loquacious or pithy (opposite of whether it was short or long to begin with.) """

    assert type(is_long) == bool, (f"Expected is_long to be a boolean, but got {is_long}",review,is_long)
    if is_long:
        assert len(review) > length_threshold, f"Review is too short: {len(review)}"
    flipped_length = "shorter" if is_long else "longer"
    counterfactual_prompt = f"""{review}\n\n[Adjust this review so it's {flipped_length}, but change *nothing* else.]
"""
    
    messages=[
                {
                    "role": "user",
                    "content": counterfactual_prompt,
                },
            ]
    return messages

# create each line of the batch input file
def write_input_jsonl(df, batch_input_filename):
    batch_input = []
    for idx, row in df.iterrows():
        messages = review_messages(row['text'], row['is_long'])
        batch_input.append({
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "messages": messages,
                "temperature": 0.7
            }
        })
    # save the batch input file as jsonl
    with open(batch_input_filename, 'w') as f:
        for item in batch_input:
            json.dump(item, f)  # Properly escape quotes and special characters
            f.write("\n")  # Ensure each JSON object is on a new line

# sample the imdb dataset
sampled_imdb = imdb.sample(sample_size, random_state=42)

# create the batch input file with a random identifier
file_id = np.random.randint(0, 10000)
batch_input_filename = f"../data/batch_api/imdb_batch_input_{file_id}_{sample_size}.jsonl"
print(batch_input_filename)

# write the batch input file
write_input_jsonl(sampled_imdb, batch_input_filename)

###################### Submit to OpenAI API #####################

# Upload the batch file
batch_input_file = client.files.create(
    file=open(batch_input_filename, "rb"),
    purpose="batch"
)

# Create the batch job
batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description":"test run"
    }
)
print(batch)

# Retrieve and print the details of the batch
time.sleep(15)
status = client.batches.retrieve(batch.id).status
print(status)

# Every 5 seconds, check the status of the batch, and retrieve if "completed"
start = time.time()
while status != "completed":
    time.sleep(10)
    status = client.batches.retrieve(batch.id).status
    print(f"Status: {status}. Time elapsed: {np.round((time.time() - start)/60,1)} minutes. Progress: {100*np.round(client.batches.retrieve(batch.id).request_counts.completed/sample_size,3)}%")
    
# create the batch output file
batch_output_filename = f"../data/batch_api/imdb_batch_output_{file_id}_{sample_size}.jsonl"
print(batch_output_filename)

# Retrieve the output file and write it to a local file
content = client.files.content(client.batches.retrieve(batch.id).output_file_id)
content.write_to_file(batch_output_filename)

print(f"Batch output file saved as {batch_output_filename}")

################### Merge and save dataframe ####################

# Process the outputs and merge them with the original DataFrame
custom_ids = []
contents = []

# Read and parse the JSON lines
with open(batch_output_filename, 'r') as file:
    for line in file:
        data = json.loads(line)
        custom_id = data.get('custom_id')
        content = data['response']['body']['choices'][0]['message']['content']
        
        custom_ids.append(int(custom_id))
        contents.append(content)

# Create a DataFrame from the extracted data
rewrites = pd.DataFrame({
    'sample_id': custom_ids,
    'rewritten_review': contents
})

# Merge the DataFrames
merged_df = pd.merge(sampled_imdb, rewrites, left_index=True, right_on='sample_id')

# Display the merged DataFrame
merged_df.head()

# Save the DataFrame to a CSV file
df_output_filename = f"../data/batch_api/imdb_rewrites_{file_id}_{sample_size}.csv"
merged_df.to_csv(df_output_filename, index=False)
print(f"Data saved to {df_output_filename}")