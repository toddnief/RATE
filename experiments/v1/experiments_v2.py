import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import get_unedited_logprobs, get_edited_logprobs, get_patching_weights_nnsight
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pathlib import Path
import scipy.stats as stats


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "config/"

with open(CONFIG_DIR / 'config.json', 'r') as f:
    config_global = json.load(f)

PROJECT_DIR = Path(config_global['project_dir'])
DATA_DIR = PROJECT_DIR / "data/"
RESULTS_DIR = PROJECT_DIR / "results/"

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3)} GB")
        print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3)} GB")
        print(f"GPU Utilization: {torch.cuda.max_memory_allocated(0) / (1024 ** 3)} GB")


def load_model(model_path, weights_path=None, device=device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict['state'])
    model = model.to(device)
    model.eval()
    return model


def truncate(text, max_len=2000, truncate_left=True):
    if truncate_left:
        return text[-max_len:] if len(text) > max_len else text
    else:
        return text[:max_len] if len(text) > max_len else text


def calculate_test_statistic(edited_interventional_likelihood, unedited_interventional_likelihood, edited_conditional_likelihood, unedited_conditional_likelihood):
    return edited_interventional_likelihood - unedited_interventional_likelihood - edited_conditional_likelihood + unedited_conditional_likelihood


def rlhf_experiment(model, tokenizer, edited_model, dataset):
    results = []
    for _, example in tqdm(dataset.iterrows(), leave=False):
        base_prompt = truncate(example['non_instruct_prompt'], truncate_left=True)
        base_prompt += '\n'  # Prevent tokenizer issues when calculating logprobs

        edited_interventional_likelihood = get_unedited_logprobs(edited_model, tokenizer, base_prompt, example['interventional_query_completion']).sum()
        edited_conditional_likelihood = get_unedited_logprobs(edited_model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        unedited_interventional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['interventional_query_completion']).sum()
        unedited_conditional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        test_statistic = calculate_test_statistic(edited_interventional_likelihood, unedited_interventional_likelihood, edited_conditional_likelihood, unedited_conditional_likelihood)

        results.append({
            'edited_interventional_likelihood': edited_interventional_likelihood.item(),
            'edited_conditional_likelihood': edited_conditional_likelihood.item(),
            'unedited_interventional_likelihood': unedited_interventional_likelihood.item(),
            'unedited_conditional_likelihood': unedited_conditional_likelihood.item(),
            'test_statistic': test_statistic.item()
        })

    return results


def aa_experiment(model, tokenizer, dataset, direction, weight):
    results = []
    for _, example in tqdm(dataset.iterrows(), leave=False):
        # TODO: figure out generalization for models with different special tokens
        # Use new line to prevent tokenizer issues when calculating logprobs
        base_prompt = example['review'] + example['next_text'] + '\n'

        # TODO: source_prompt is a confusing variable name
        # sample a random review with opposite sentiment from the example
        # TODO: probably a generalization for this across datasets...
        # source_sample = dataset[dataset.label != example['label']].sample(n=1)
        # source_prompt = source_sample["baseline_query"].iloc[0]
        # # TODO: do we actually want to do this or?
        # weights = get_patching_weights_nnsight(model, base_prompt, source_prompt, direction)

        # Note: baseline example is positive sentiment so edited completion is negative sentiment
        if example['label'] == 1:
            weight = -weight

        edited_interventional_likelihood = get_edited_logprobs(model, tokenizer, base_prompt, example['interventional_query_completion'], weight, direction, hidden_state_path="transformer.ln_f").sum()
        edited_conditional_likelihood = get_edited_logprobs(model, tokenizer, base_prompt, example['conditional_query_completion'], weight, direction, hidden_state_path="transformer.ln_f").sum()

        unedited_interventional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['interventional_query_completion']).sum()
        unedited_conditional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        test_statistic = calculate_test_statistic(edited_interventional_likelihood, unedited_interventional_likelihood, edited_conditional_likelihood, unedited_conditional_likelihood)

        results.append({
            'edited_interventional_likelihood': edited_interventional_likelihood.item(),
            'edited_conditional_likelihood': edited_conditional_likelihood.item(),
            'unedited_interventional_likelihood': unedited_interventional_likelihood.item(),
            'unedited_conditional_likelihood': unedited_conditional_likelihood.item(),
            'test_statistic': test_statistic.item()
        })
    return results


def sft_experiment(model, tokenizer, edited_model, dataset):
    results = []
    for _, example in tqdm(dataset.iterrows(), leave=False):

        # Model is finetuned to output positive sentiment so skip any examples edited to output negative sentiment
        if example['flipped_sentiment'] == "negative":
            continue

        base_prompt = example['review'] + example['next_text'] + '\n'

        edited_interventional_likelihood = get_unedited_logprobs(edited_model, tokenizer, base_prompt, example['interventional_query_completion']).sum()
        edited_conditional_likelihood = get_unedited_logprobs(edited_model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        unedited_interventional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['interventional_query_completion']).sum()
        unedited_conditional_likelihood = get_unedited_logprobs(model, tokenizer, base_prompt, example['conditional_query_completion']).sum()

        test_statistic = calculate_test_statistic(edited_interventional_likelihood, unedited_interventional_likelihood, edited_conditional_likelihood, unedited_conditional_likelihood)

        results.append({
            'edited_interventional_likelihood': edited_interventional_likelihood.item(),
            'edited_conditional_likelihood': edited_conditional_likelihood.item(),
            'unedited_interventional_likelihood': unedited_interventional_likelihood.item(),
            'unedited_conditional_likelihood': unedited_conditional_likelihood.item(),
            'test_statistic': test_statistic.item()
        })

    return results


def analyze_results(results, confidence_level=0.95):
    test_statistic_mean = results['test_statistic'].mean()
    sem = results['test_statistic'].sem()
    
    # Calculate the confidence interval
    n = len(results['test_statistic'])
    h = sem * stats.t.ppf((1 + confidence_level) / 2., n-1)
    confidence_interval = (test_statistic_mean - h, test_statistic_mean + h)
    
    return test_statistic_mean, confidence_interval


def run(model, tokenizer, datapath, model_name, experiment, edited_model=None, direction=None, weight=None, confidence_level=.95):
    with open(DATA_DIR / datapath, "r") as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset)

    if experiment == "rlhf":
        results = rlhf_experiment(model, tokenizer, edited_model, df)
    if experiment == "aa":
        results = aa_experiment(model, tokenizer, df, direction, weight)
    if experiment == "sft":
        results = sft_experiment(model, tokenizer, edited_model, df)

    filename = f"{experiment}_experiment_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(RESULTS_DIR / filename, 'w') as file:
        json.dump(results, file, indent=4)

    results = pd.DataFrame(results)
    print(results.describe())

    print(f"Results have been saved to {filename}")

    test_statistic_mean, confidence_interval = analyze_results(results, confidence_level)

    print(f"Mean test statistic: {test_statistic_mean}")
    print(f"{confidence_level*100}% Confidence Interval: {confidence_interval}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, default="rlhf", help="The experiment to run. Options are 'rlhf',  'aa', and 'sft'")

    args = parser.parse_args()

    experiment = args.experiment

    if experiment == "rlhf":
        with open(CONFIG_DIR / 'config_rlhf.json', 'r') as f:
            config = json.load(f)

        datapath = DATA_DIR / config['data_file']

        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
        tokenizer.pad_token = tokenizer.eos_token

        # Load original model
        print("Loading original model...")
        model_config = config['original_model']
        model = load_model(model_path=model_config['model_path'], weights_path=model_config['weights_path'], device=device)

        # Load and run experiments for finetuned models
        for finetuned_model in config['finetuned_models']:
            print(f"Loading {finetuned_model['name']} model...")
            edited_model = load_model(model_path=finetuned_model['model_path'], weights_path=finetuned_model['weights_path'], device=device)
            
            run(model, tokenizer, datapath, edited_model=edited_model, model_name=finetuned_model['name'], experiment="rlhf")
    elif experiment == "aa":
        from nnsight import LanguageModel

        # TODO: standardize config structure a bit more...
        with open(CONFIG_DIR / 'config_aa.json', 'r') as f:
            config = json.load(f)

        datapath = DATA_DIR / config['data_file']

        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        with open(config['direction_path'], 'r') as f:
            direction_raw = json.load(f)
        direction = torch.tensor(direction_raw)
        direction = direction.to(device)

        print(f"Loading original model {config['model_name']}...")
        model = LanguageModel(config['model_name'], device_map=device)
        
        run(model, tokenizer, datapath, model_name=config['model_name'], experiment="aa", direction=direction, weight=config['weight'])
    elif experiment == "sft":
        with open(CONFIG_DIR / 'config_sft.json', 'r') as f:
            config = json.load(f)

        datapath = DATA_DIR / config['data_file']

        tokenizer = AutoTokenizer.from_pretrained(config['original_model_name'])
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading original model {config['original_model_name']}...")
        model = load_model(model_path=config['original_model_name'], device=device)

        print(f"Loading edited model {config['sft_model_name']}...")
        edited_model = load_model(model_path=config['sft_model_name'], device=device)

        run(model, tokenizer, datapath, edited_model=edited_model, model_name=config['original_model_name'], experiment="sft")
