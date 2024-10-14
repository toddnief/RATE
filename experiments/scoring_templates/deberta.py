def get_deberta_reward(outputs):
    return outputs.logits[0].detach().item()
