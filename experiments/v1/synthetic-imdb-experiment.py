# Load imdb for examples
import pandas as pd
from sklearn.metrics import mutual_info_score
import numpy as np

import sys

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

################# Load the merged dataframe #################

# When this file is called, the filename of the merged DataFrame is passed as an argument

file_id = sys.argv[1]
sample_size = int(sys.argv[2])
merged_df_filename = f"../data/batch_api/imdb_rewrites_{file_id}_{sample_size}.csv"

# Load the saved DataFrame
merged_df = pd.read_csv(merged_df_filename)
print(merged_df.head())
assert sample_size == merged_df.shape[0]
print("Sample size:", sample_size)

length_threshold = 970.0

################# Setup the Sentiment Classifier #################

sentiment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment_score(review):
    inputs = sentiment_tokenizer(review, return_tensors="pt", truncation=True)

    if inputs['input_ids'].shape[1] > 512:
        chunk_size = 512
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        chunked_probs = []
        for i in range(0, input_ids.size(0), chunk_size):
            input_chunk = input_ids[i:i + chunk_size].unsqueeze(0)
            attention_chunk = attention_mask[i:i + chunk_size].unsqueeze(0)
            with torch.no_grad():
                logits = sentiment_model(input_ids=input_chunk, attention_mask=attention_chunk).logits
                probs = F.softmax(logits, dim=-1)
            chunked_probs.append(probs)

        # Compute geometric mean of probabilities across chunks
        mean_probs = torch.exp(torch.mean(torch.log(torch.stack(chunked_probs)), dim=0))

        # Convert the mean probabilities back to logits
        logits = torch.log(mean_probs)
    else:
        with torch.no_grad():
            logits = sentiment_model(**inputs).logits

    positive_logit = logits[0, 1].item()  # Directly return the logit for positive sentiment

    return positive_logit

# Apply the sentiment classifier to both the original and rewritten reviews
merged_df['orig_reward'] = merged_df['text'].apply(get_sentiment_score)
merged_df['rewrite_reward'] = merged_df['rewritten_review'].apply(get_sentiment_score)

# save the merged_df with the sentiment scores
merged_df.to_csv(f"../data/batch_api/imdb_scored_{file_id}_{sample_size}.csv", index=False)

# ### Strategy for Creating Gradual Correlation
# Bin Text Lengths

# - Texts are divided into two bins based on their length:
#         - Texts longer than 1200 characters
#         - Texts shorter than or equal to 1200 characters
#     - Note: Given a truncation at 4000 characters, each bin is equally likely to be sampled when reviewing either positive or negative samples, ensuring an initial uniform distribution across bins.

# Initial Distribution

#     - The initial distribution across the two bins is modeled as a Bernoulli distribution with p=0.5, indicating no initial correlation between text length and sentiment:
#         - P(long∣positive)=0.5
#         - P(long∣negative)=0.5

# Gradual Increase in Correlation

#     - A sequence of DataFrames will be generated to represent a gradually increasing correlation between long reviews and positive sentiments:
#         - For each experiment iteration i from 1 to 10:
#             - Initialize the ith DataFrame.
#             - For each sample j from 1 to num_samples:
#                 - Sample the sentiment sj​ from Bernoulli(0.5), where 1 represents a positive sentiment and 0 represents a negative sentiment.
#                 - Conditionally sample the review length:
#                     - If sj​ is positive, sample from the long bin with P(long∣sj=positive)=0.5+0.05*i
#                     - If sj​ is negative, sample from the long bin with P(long∣sj=negative)=0.5−0.05*i

# Target Distribution for Perfect Correlation

#     - In the final step (i=10), achieve perfect correlation:
#         - P(long∣positive)=1: All long reviews will be positive.
#         - P(long∣negative)=0: No long reviews will be negative.

# Parameters
num_dfs = 10
num_samples = sample_size // num_dfs
assert num_samples >= 10, "Number of samples per DataFrame is too low"
print("Number of samples per DataFrame:", 2*num_samples)

# Split the dataset into long and short texts
long_texts = merged_df[merged_df['is_long']]
short_texts = merged_df[~merged_df['is_long']]
print(long_texts.shape, short_texts.shape)

# Initialize list to hold the DataFrames
correlated_dataframes = []

# Generate a sequence of DataFrames with increasing correlation
for i in range(1, num_dfs + 1):
    # Define probabilities for long reviews based on the current iteration
    p_long_positive = 0.5 + 0.05 * i
    p_long_negative = 0.5 - 0.05 * i

    # Determine how many long texts should be selected for each sentiment
    num_long_positive = np.random.binomial(num_samples, p_long_positive)
    num_long_negative = np.random.binomial(num_samples, p_long_negative)

    # Sample texts based on the computed probabilities
    positive_samples = pd.concat([
        long_texts[long_texts['label'] == 1].sample(num_long_positive, replace=False),
        short_texts[short_texts['label'] == 1].sample(num_samples - num_long_positive, replace=False)
    ])

    negative_samples = pd.concat([
        long_texts[long_texts['label'] == 0].sample(num_long_negative, replace=False),
        short_texts[short_texts['label'] == 0].sample(num_samples - num_long_negative, replace=False)
    ])

    # Combine positive and negative samples
    combined_samples = pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)

    # Append the DataFrame to the list
    correlated_dataframes.append(combined_samples)

# `correlated_dataframes` now contains the DataFrames with increasing correlation

# provide header, summary statistics, and a few examples for each dataframe
for i, df in enumerate(correlated_dataframes):
    print(f"DataFrame {i + 1}")
    print(df['label'].value_counts())
    print(df['text'].str.len().describe())
    print("\n")

# check the mutual information between the label and the text length for each of the generated dataframes
mi_values = []

for i, df in enumerate(correlated_dataframes):
    df['text_length'] = df['text'].str.len()
    df['is_long'] = df['text_length'] > length_threshold
    mi = mutual_info_score(df['label'], df['is_long'])
    mi_values.append(mi)

print("mi_values", mi_values)

# compute the entropy of each of the generated dataframes
entropy_values = []

for i, df in enumerate(correlated_dataframes):
    entropy = df['label'].value_counts(normalize=True).map(lambda x: x * np.log(x)).sum()
    entropy_values.append(entropy)

print("entropy_values", entropy_values)

# calculate the correlation coefficient between the label and the text length for each of the generated dataframes
correlation_values = []

for i, df in enumerate(correlated_dataframes):
    df['text_length'] = df['text'].str.len()
    correlation = df[['label', 'text_length']].corr().iloc[0, 1]
    correlation_values.append(correlation)

print("correlation_values", correlation_values)

#################### Compute effect sizes ####################

def calculate_ate(dataset):
    do_w_1 = 0
    do_w_0 = 0
    w_1 = 0
    w_0 = 0
    w_1_count = 0
    for example in dataset:
        if example["w_original"]:
            do_w_1 += example["reward"]["original"]
            do_w_0 += example["reward"]["rewrite"]
            w_1_count += 1

            w_1 += example["reward"]["original"]
        else:
            do_w_1 += example["reward"]["rewrite"]
            do_w_0 += example["reward"]["original"]

            w_0 += example["reward"]["original"]

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
        if example["w_original"]:
            do_w_1_diff = example["reward"]["original"] - do_w_1_mean
            do_w_0_diff = example["reward"]["rewrite"] - do_w_0_mean
        else:
            do_w_1_diff = example["reward"]["rewrite"] - do_w_1_mean
            do_w_0_diff = example["reward"]["original"] - do_w_0_mean

        do_w_1_ssd += do_w_1_diff**2
        do_w_0_ssd += do_w_0_diff**2

    do_w_1_var = do_w_1_ssd / (n - 1)
    do_w_0_var = do_w_0_ssd / (n - 1)

    return ATE, do_w_1_var, do_w_0_var, correlational_effect

def calculate_effects(df):
    # use todd's formatting to organize the data from merged_df
    dataset = []
    for idx, row in df.iterrows():
        dataset.append(
            {
                "w_original": row['is_long'],
                "completions": {
                    "original": row['text'],
                    "rewrite": row['rewritten_review'],
                },
                "reward": {
                    "original": row['orig_reward'],
                    "rewrite": row['rewrite_reward'],
                },
            }
        )

    ATE, do_w_1_var, do_w_0_var, correlational_effect = calculate_ate(dataset)
    return ATE, do_w_1_var, do_w_0_var, correlational_effect

ATE, do_w_1_var, do_w_0_var, correlational_effect = calculate_effects(merged_df)
print("ATE on merged df:", ATE)
print("Variance of do_w_1:", do_w_1_var)
print("Variance of do_w_0:", do_w_0_var)
print("Correlational effect:", correlational_effect)

# Calculate the effects for each of the generated dataframes

ATE_values = []
do1_variations = []
do0_variations = []
correlational_effect_values = []

for df in correlated_dataframes:
    ATE, do_w_1_var, do_w_0_var, correlational_effect = calculate_effects(df)
    ATE_values.append(ATE)
    do1_variations.append(do_w_1_var)
    do0_variations.append(do_w_0_var)
    correlational_effect_values.append(correlational_effect)

#######################  Save the results  #######################

# Save the results to a CSV file
results_df = pd.DataFrame({
    "ATE": ATE_values,
    "do1 variation": do1_variations,
    "do0 variation": do0_variations,
    "correlational_effect": correlational_effect_values,
    "mi": mi_values,
    "entropy": entropy_values,
    "correlation": correlation_values
})

results_df.to_csv(f"../data/batch_api/imdb_results_{file_id}_{sample_size}.csv", index=False)

print("Results saved successfully")