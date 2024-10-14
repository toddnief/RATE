import json
import torch
# from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import get_unedited_logprobs, get_edited_logprobs, get_patching_weights_nnsight
from argparse import ArgumentParser
from tqdm import tqdm
import os
import random
import pandas as pd
from rlhf_models import load_model
from datetime import datetime

# TODO: make this robust, use paths, blah blah...
# PROMPTS_PATH = "../data/prompt_completions_gpt4.json"


VEITCH_DIR = "/net/projects/veitch/prompt_distributions/"
DATA_DIR = VEITCH_DIR + "data/"
RESULTS_DIR = VEITCH_DIR + "results/"

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  
torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weight", type=float, default=20)
    parser.add_argument("--model_name", type=str, default="gemma")

    results = {}
    for j, example in tqdm(df.iterrows(), leave=False):
        # Use start of turn token to prevent tokenizer issues when calculating logprobs

        # TODO: figure out generalization for models with different special tokens
        base_prompt = example['review'] + example['next_text'] + model.tokenizer.additional_special_tokens[0]

        # TODO: source_prompt is a confusing variable name
        # sample a random review with opposite sentiment from the example
        # TODO: probably a generalization for this across datasets...
        source_sample = df[df.label != example['label']].sample(n=1)
        source_prompt = source_sample["baseline_query"].iloc[0]
        weights = get_patching_weights_nnsight(model, base_prompt, source_prompt, direction)

        edited_interventional_likelihood = get_edited_logprobs(model, base_prompt, example['interventional_query_completion'], weights, direction).sum()
        edited_conditional_likelihood = get_edited_logprobs(model, base_prompt, example['conditional_query_completion'], weights, direction).sum()

        unedited_interventional_likelihood = get_unedited_logprobs(model, toeknizer, base_prompt, example['interventional_query_completion']).sum()
        unedited_conditional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        test_statistic = edited_interventional_likelihood - unedited_interventional_likelihood - edited_conditional_likelihood + unedited_conditional_likelihood

        results[i].append({
            'edited_interventional_likelihood': edited_interventional_likelihood.item(),
            'edited_conditional_likelihood': edited_conditional_likelihood.item(),
            'unedited_interventional_likelihood': unedited_interventional_likelihood.item(),
            'unedited_conditional_likelihood': unedited_conditional_likelihood.item(),
            'test_statistic': test_statistic.item()
        })

    # TODO: Create a consistent filename for this
    with open('test_results.json', 'w') as file:
        json.dump(results, file, indent=4)

    print("Results have been saved to test_results.json")

def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3)} GB")
        print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3)} GB")
        print(f"GPU Utilization: {torch.cuda.max_memory_allocated(0) / (1024 ** 3)} GB")


    weight = args.weight
    model_name = args.model_name
    
    with open(os.path.join(data_dir, data_file), "r") as f:
        dataset = json.load(f)


def run_rlhf_experiment(model, tokenizer, edited_model, dataset):
    df = pd.DataFrame(dataset)

    if model_name == "gemma":
        model_name = 'google/gemma-2b-it'
        model = LanguageModel(model_name, device_map=device)
    elif model_name == "gpt2-imdb":
        model_name = 'lvwerra/gpt2-imdb'
        model = LanguageModel(model_name, device_map=device)
        df = df[df.label == 0]

    # TODO: Maybe have a more principled way of loading directions
    # saved_directions = f"/net/projects/veitch/geometry_llms/directions/intervention/sentiment_{model_name.split('/')[-1]}.pt"
    # directions = torch.load(saved_directions)

    direction_file = "/net/projects/veitch/prompt_distributions/directions/das_simple_train_ADJ_layer12.json"
    with open(direction_file, 'r') as f:
        data = json.load(f)
    direction = torch.tensor(data).to(device)
    desired_magnitude = torch.norm(direction)

    direction = torch.randn(768).to(device) * desired_magnitude / torch.norm(direction)
    directions = [direction]

    '''
    GemmaForCausalLM(
        (model): GemmaModel(
            (embed_tokens): Embedding(256000, 2048, padding_idx=0)
            (layers): ModuleList(
            (0-17): 18 x GemmaDecoderLayer(
                (self_attn): GemmaSdpaAttention(
                (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
                (k_proj): Linear(in_features=2048, out_features=256, bias=False)
                (v_proj): Linear(in_features=2048, out_features=256, bias=False)
                (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
                (rotary_emb): GemmaRotaryEmbedding()
                )
                (mlp): GemmaMLP(
                (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
                (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
                (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
                (act_fn): PytorchGELUTanh()
                )
                (input_layernorm): GemmaRMSNorm()
                (post_attention_layernorm): GemmaRMSNorm()
            )
            )
            (norm): GemmaRMSNorm()
        )
        (lm_head): Linear(in_features=2048, out_features=256000, bias=False)
        (generator): WrapperModule()
        )
    '''

    '''
    GPT2LMHeadModel(
        (transformer): GPT2Model(
            (wte): Embedding(50257, 768)
            (wpe): Embedding(1024, 768)
            (drop): Dropout(p=0.1, inplace=False)
            (h): ModuleList(
            (0-11): 12 x GPT2Block(
                (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (attn): GPT2Attention(
                (c_attn): Conv1D()
                (c_proj): Conv1D()
                (attn_dropout): Dropout(p=0.1, inplace=False)
                (resid_dropout): Dropout(p=0.1, inplace=False)
                )
                (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): GPT2MLP(
                (c_fc): Conv1D()
                (c_proj): Conv1D()
                (act): NewGELUActivation()
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            )
            (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (lm_head): Linear(in_features=768, out_features=50257, bias=False)
        (generator): WrapperModule()
        )
    '''


    for i, direction in enumerate(directions):
        results[i] = []
        print(f"##### Running direction {i} #####")
        for j, example in tqdm(df.iterrows(), leave=False):
            if model_name == 'google/gemma-2b-it':
                # Use start of turn token to prevent tokenizer issues when calculating logprobs
                base_prompt = example['review'] + example['next_text'] + model.tokenizer.additional_special_tokens[0]
            else:
                base_prompt = example['review'] + example['next_text'] + '\n'

            # TODO: source_prompt is a confusing variable name
            # sample a random review with opposite sentiment from the example
            # source_sample = df[df.label != example['label']].sample(n=1)
            # source_prompt = source_sample["baseline_query"].iloc[0]
            # TODO: Add something here to make this an option
            # weights = get_patching_weights_nnsight(model, base_prompt, source_prompt, direction)

            weights = -20

        # load the tokenizer. this is the same for all models
        tokenizer = AutoTokenizer.from_pretrained('/net/scratch/dpo/Step1_SFT/Step1_SFT_Antrophic_Pythia28')
        tokenizer.pad_token = tokenizer.eos_token

        ### Original Unedited Model ###
        print("Loading original model...")
        weight_path = '/net/projects/veitch/llm_alignment/direct-preference-optimization/.cache/garbacea/anthropic_dpo_pythia28_2024-01-27_20-42-34_616113/LATEST/policy.pt'
        model = load_model(model_path='EleutherAI/pythia-2.8b', weights_path=weight_path, device=device)

        #### Best Finetuned Model ####
        print("Loading RLHF'd model...")
        weight_path = '/net/projects/veitch/llm_alignment/direct-preference-optimization/.cache/garbacea/anthropic_ipo_pythia28_baseline_hh_subset_best_and_worst_of_8_beta00275482094_combined_loss_alpha0005_separate_loss_logs_seed5_2024-05-01_18-01-07_469089/step-19968/policy.pt'
        edited_model = load_model(model_path='EleutherAI/pythia-2.8b', weights_path=weight_path, device=device)

        run_rlhf_experiment(model, tokenizer, edited_model, dataset)

    elif experiment == "activation-addition":
        data_file = "prompt_completions_gpt-4-1106-preview_subset300_seed123_numtokens40_next1_temp0.7_job23632.json"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results have been saved to {filename}")
