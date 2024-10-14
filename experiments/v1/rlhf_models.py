import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import gc
import tqdm


# the function to load the language model
def load_model(model_path, weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict['state'])
    model = model.to(device)
    model.eval()
    return model

# load the tokenizer. this is the same for all models
tokenizer = AutoTokenizer.from_pretrained('/net/scratch/dpo/Step1_SFT/Step1_SFT_Antrophic_Pythia28')
tokenizer.pad_token = tokenizer.eos_token

# some arguments for model generation
gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}


if __name__ == '__main__':

    # specify the device
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  


    ### Original Unedited Model ###
    weight_path = '/net/projects/veitch/llm_alignment/direct-preference-optimization/.cache/garbacea/anthropic_dpo_pythia28_2024-01-27_20-42-34_616113/LATEST/policy.pt'

    #### Best Finetuned Model ####
    weight_path = '/net/projects/veitch/llm_alignment/direct-preference-optimization/.cache/garbacea/anthropic_ipo_pythia28_baseline_hh_subset_best_and_worst_of_8_beta00275482094_combined_loss_alpha0005_separate_loss_logs_seed5_2024-05-01_18-01-07_469089/step-19968/policy.pt'

    #### Adequate Finetuned MOdel ####
    weight_path = '/net/projects/veitch/llm_alignment/direct-preference-optimization/.cache/garbacea/anthropic_ipo_pythia28_baseline_hh_subset_best_and_worst_of_8_beta00275482094_seed_1_2024-05-01_22-36-18_104383/step-19968/policy.pt'

    # load the check point and move model to gpus
    model = load_model(model_path='EleutherAI/pythia-2.8b', weights_path=weight_path, device=device)
    NUM_OF_GPUS = torch.cuda.device_count()
    if NUM_OF_GPUS > 1:
        print(f"Using {NUM_OF_GPUS} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    print('Model loaded.')

