import torch

NEG = -1e9  # score given to illegal actions

def masked_log_probs(logits, mask):
    """Caculates log probability of every action, making masked actions NEG"""
    maskedLogits = torch.where(mask, logits, NEG)
    return torch.log_softmax(maskedLogits, dim=1)

def sample(log_p, gen=None):
    """Pick one action per row at random"""
    x = log_p.exp()
    return torch.multinomial(x, 1, generator=gen).squeeze(1)

def pick(log_p, actions):
    """Finds the log(p) of each chosen action per row"""
    return log_p.gather(1, actions.unsqueeze(1)).squeeze(1)

def entropy(log_p):
    """Calculates entropy of each row (sum of p * logp)"""
    p = log_p.exp()
    mul = p * log_p
    return -mul.sum(1)