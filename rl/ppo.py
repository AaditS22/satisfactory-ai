import torch
import numpy as np
from rl.dist import masked_log_probs, pick, entropy

def ppo_losses(new_logp, old_logp, adv, new_value, ret, clip=0.2):
    """Calculates two PPO losses for one batch of B steps"""
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
    value_loss = (new_value - ret).square().mean()
    return policy_loss, value_loss

def normalise(adv):
    """Normalizes advantages to have a mean of 0 and var of 1"""
    return (adv - adv.mean()) / (adv.std() + 1e-8)
 
def to_tensors(data, device):
    """Turns numpy arrays to tensors made for the given device"""
    out = {}
    for k, v in data.items():
        t = torch.as_tensor(v, device=device)
        if k == "masks":
            t = t.bool()
        elif k == "actions":
            t = t.long()
        else:
            t = t.float()
        out[k] = t
    return out
 
def health(new_logp, old_logp, clip):
    """Measures how far the probabilities have moved from the page's network, and the fraction of steps that have exceeded the clipping ratio"""
    with torch.no_grad():
        log_ratio = new_logp - old_logp
        ratio = log_ratio.exp()
        kl = ((ratio - 1) - log_ratio).mean().item()
        frac = ((ratio - 1).abs() > clip).float().mean().item()
    return kl, frac
 
def update(policy, optimizer, buffer, obs_to_torch, epochs=4, minibatch=512, clip=0.2,
           vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, rng=None):
    """Learn from one full buffer page, doing 'ephocs' passes, each split into random groups of `minibatch` steps"""
    rng = rng or np.random.default_rng()
    device = next(policy.parameters()).device
    total = buffer.T * buffer.N
    log = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
 
    for epoch in range(epochs):
        order = rng.permutation(total)
        for start in range(0, total, minibatch):
            idx = order[start:start + minibatch]
            obs, data = buffer.get(idx)
            obs = obs_to_torch(obs, device)
            d = to_tensors(data, device)
 
            logits, new_value = policy(obs)
            log_p = masked_log_probs(logits, d["masks"])
            new_logp = pick(log_p, d["actions"])
            ent = entropy(log_p).mean()
            adv = normalise(d["adv"])
            pl, vl = ppo_losses(new_logp, d["logp"], adv, new_value, d["ret"], clip)
            loss = pl + vf_coef * vl - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
 
            kl, frac = health(new_logp, d["logp"], clip)
            log["policy_loss"].append(pl.item())
            log["value_loss"].append(vl.item())
            log["entropy"].append(ent.item())
            log["approx_kl"].append(kl)
            log["clip_frac"].append(frac)
 
    return {k: float(np.mean(v)) for k, v in log.items()}

def linear_lr(r, rounds, lr_max):
    """Learning rate for round r of a run that has played 'r' rounds"""
    return lr_max * (1.0 - (r / rounds))