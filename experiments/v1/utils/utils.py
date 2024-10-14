import torch
# import tiktoken

def process_reviews(dataset, tokenizer, num_tokens=40, num_next_tokens=10):
    """Assumes each sample in the dataset has a 'text' and 'label' key.
    Accepts:
    - dataset (list): a list of dictionaries, each containing a 'text' and 'label' key.
    - tokenizer (tiktoken BPE tokenizer)
    Returns a list of positive and negative reviews, each truncated to num_tokens."""
    pos_prompts, neg_prompts = [], []
    for sample in dataset:
        # Encode and truncate the text to the specified number of tokens
        tokens = tokenizer.encode(sample["text"])
        truncated_tokens = tokens[:num_tokens]  # truncate to the first num_tokens tokens
        truncated_text = tokenizer.decode(truncated_tokens)  # decode back to text

        # Get the next set of tokens after the truncated part
        next_tokens_start = num_tokens
        next_tokens_end = num_tokens + num_next_tokens
        if next_tokens_end <= len(tokens):  # Ensure there are enough tokens
            next_tokens = tokens[next_tokens_start:next_tokens_end]
            next_text = tokenizer.decode(next_tokens)
        else:
            next_text = ""  # Not enough tokens for 'next_text'
        
        # Append based on the label
        if sample['label'] == 1:
            pos_prompts.append((truncated_text, next_text))
        else:
            neg_prompts.append((truncated_text, next_text))
    
    return pos_prompts, neg_prompts


def get_generation_probs(model, tokenizer, prompt, device="cuda:0", distributions=False):
    """Generates text from a prompt and returns the generation probabilities for each token.
    Use for standard huggingface models.
    """
    templated_prompt = [
        {"role": "user", "content": prompt}
    ]
    prompt_ids = tokenizer.apply_chat_template(templated_prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(prompt_ids, max_new_tokens=1000, do_sample=True)
    generated_only_ids = generated_ids[:, prompt_ids.shape[1]:]

    logits = model.forward(generated_ids).logits
    logits_generated = logits[:, prompt_ids.shape[1]:, :]
    generation_probs_distribution = torch.nn.functional.softmax(logits_generated, dim=-1)

    if distributions:
        return generation_probs_distribution
    else:
        generation_probs = generation_probs_distribution[0, torch.arange(generated_only_ids.shape[1]), generated_only_ids[0, :]]
        return generation_probs

############################################ Get logprobs for addition via nnsight ############################################

def combine_prompts_gpt4(prompt1, prompt2):
    # Get the tokenizer for GPT-4
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    # Encode the prompts
    tokens1 = tokenizer.encode(prompt1)
    tokens2 = tokenizer.encode(prompt2)
    # Combine the tokens
    tokens = tokens1 + tokens2
    # Decode the tokens
    return tokenizer.decode(tokens)

def get_completion_logprobs(logprobs, tokenizer, prompt, completion, device="cuda:0"):
    """
    Get the log probabilities of the completion given the prompt using the model.

    Parameters:
    - model (nnsight.LanguageModel): The language model object.
    - prompt (str): The initial text input for text generation.
    - completion (str): The text to be completed.

    Returns:
    - torch.Tensor: A tensor of the log probabilities for the predicted next token IDs.
    """
    completion_logprobs = logprobs[:, len(tokenizer(prompt)['input_ids']):, :]
    completion_token_ids = tokenizer(completion, return_tensors='pt')['input_ids'][:,1:].to(device) # remove the first token <bos>
    completion_token_logprobs = completion_logprobs.gather(2, completion_token_ids.unsqueeze(-1))
    return completion_token_logprobs

def get_edited_logprobs(model, tokenizer, prompt, completion, weights, direction, hidden_state_path, edit_method="addition", device="cuda:0"):
    """
    Adjusts the hidden states of a language model by applying a scaled directional vector,
    then predicts the log probabilities of the next token. Compatible with Mistral and Gemma models.

    Parameters:
    - model (nnsight.LanguageModel): The language model object.
    - prompt (str): The initial text input for text generation.
    - completion (str): The text to be completed.
    - weight (float): The scaling factor for the directional vector.
    - direction (torch.Tensor): The vector used to alter the model's hidden states.

    Returns:
    - torch.Tensor: A tensor of the log probabilities for the predicted next token IDs.
    """
    # TODO: How should we actually handle spaces
    with model.trace(prompt + completion) as tracer:
        if edit_method == "addition":
            # add the direction to the hidden states
            # TODO: We should check this...and make sure this actually works for what we're doing
            hidden_states = model.transformer.ln_f.output.clone().save()
            hidden_states = hidden_states + weights * direction
        elif edit_method == "patching":
            # TODO: Implement this
            pass

        # finish running the model
        output = model.lm_head(hidden_states)
        logprobs = torch.log(torch.nn.functional.softmax(output, dim=-1)).save()
    return get_completion_logprobs(logprobs, tokenizer, prompt, completion, device=device)

def get_unedited_logprobs(model, tokenizer, prompt, completion, device="cuda:0"):
    logits = model.forward(tokenizer(prompt + completion, return_tensors='pt')['input_ids'].to(device)).logits
    logprobs = torch.log(torch.nn.functional.softmax(logits, dim=-1))
    return get_completion_logprobs(logprobs, tokenizer, prompt, completion, device=device)


############################## Compute weights corresponding to activation patching ###############################

def scalar_projection_torch(a, b):
    """
    Computes the scalar projection of tensor a onto tensor b using PyTorch. This function handles tensors of shape 
    (batchsize, seq_len, hidden_dim) for `a`
    and (batchsize, 1, hidden_dim) or (batchsize, seq_len, hidden_dim) for `b`.

    Parameters:
    - a: torch.Tensor, the tensor to be projected, shape (batchsize, seq_len, hidden_dim).
    - b: torch.Tensor, the tensor to project onto, of shape (batchsize, 1, hidden_dim) or (batchsize, seq_len, hidden_dim).

    Returns:
    - torch.Tensor: The scalar magnitude of the projection of a onto b, shape (batchsize, seq_len, 1).
    """
    # Compute the dot product across the last dimension (hidden_dim).
    dot_product = torch.sum(a * b, dim=-1, keepdim=True)  # Shape: (batchsize, seq_len, 1)

    # Compute the magnitude squared of vector `b` across the last dimension.
    magnitude_squared = torch.sum(b * b, dim=-1, keepdim=True)  # Shape: (batchsize, seq_len, 1)

    # Compute the scalar projection.
    scalar_projection = dot_product / magnitude_squared  # Shape: (batchsize, seq_len, 1)
    return scalar_projection

def compute_directional_weights(base_activations, source_activations, direction):
    """
    Computes weights for directional activation patching by comparing projections onto a direction vector.
    Uses the dot product directly due to normalization of the direction vector.

    Assumes the base and source activations have the same shape.

    Parameters:
    - base_activations: torch.Tensor, original activations, shape (batch_size, seq_length, hidden_dim).
    - source_activations: torch.Tensor, activations from a different source, shape (batch_size, seq_length, hidden_dim).
    - direction: torch.Tensor, the direction vector to project onto, shape (hidden_dim).

    Returns:
    - torch.Tensor: Weights computed from the difference in dot products, shape (batch_size, seq_length, 1).
    """
    # Ensure the base and source activations have the same shape
    assert base_activations.shape == source_activations.shape

    # Reshape normalized direction for proper broadcasting with activations of shape (batch_size, seq_length, hidden_dim)
    direction_reshaped = direction.view(1, 1, -1)

    # Compute scalar projections for both base and source activations with the direction
    scalar_base = scalar_projection_torch(base_activations, direction_reshaped)
    scalar_source = scalar_projection_torch(source_activations, direction_reshaped)

    # The weight is the difference in the scalar projections
    weight = scalar_source - scalar_base
    return weight

def adjust_len_activations(base_activations, source_activations):
    """
    Adjusts source_activations to match the size of base_activations by padding or truncating.
    
    - If source_activations is longer, it will be truncated from the left.
    - If source_activations is shorter, it will be left-padded with activations from base_activations.
    
    Parameters:
    - base_activations: torch.Tensor, shape (1, num_tokens_base_seq, hidden_dim)
    - source_activations: torch.Tensor, shape (1, num_tokens_source_seq, hidden_dim)

    Returns:
    - torch.Tensor: Adjusted source_activations with the same shape as base_activations.
    """
    base_seq_len = base_activations.shape[1]
    source_seq_len = source_activations.shape[1]

    if source_seq_len > base_seq_len:
        # Truncate source_activations from the left
        modified_source_activations = source_activations[:, -base_seq_len:, :]
    elif source_seq_len < base_seq_len:
        # Calculate the number of tokens to pad
        pad_length = base_seq_len - source_seq_len
        # Left-pad source_activations with the corresponding initial activations from base_activations
        # This is done to ensure that the weights are zero for the padded tokens
        padding = base_activations[:, :pad_length, :]
        modified_source_activations = torch.cat([padding, source_activations], dim=1)
    else:
        # If they are the same length, no adjustment is needed
        modified_source_activations = source_activations

    print(f"Modified source sequence length: {modified_source_activations.shape[1]}")

    return modified_source_activations


def get_patching_weights_nnsight(model, base_prompt, source_prompt, direction):
    """
    Computes weights for directional activation patching by comparing projections onto a direction vector.
    Uses the dot product directly due to normalization of the direction vector.

    Parameters:
    - model (nnsight.LanguageModel): The language model object.
    - base_prompt (str): The initial text input for text generation.
    - source_prompt (str): The text from a different source.
    - direction (torch.Tensor): The direction vector to project onto.

    Returns:
    - torch.Tensor: Weights computed from the difference in dot products.
    """
    with model.trace(base_prompt) as tracer:
        base_activations = model.model.norm.output.save()

    with model.trace(source_prompt) as tracer:
        source_activations = model.model.norm.output.save()

    # Compute the weights based on the base and source activations
    modified_source_activations = adjust_len_activations(base_activations, source_activations)
    weights = compute_directional_weights(base_activations, modified_source_activations, direction)

    return weights