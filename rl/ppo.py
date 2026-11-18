import torch
import numpy as np
from rl.dist import masked_log_probs, pick, entropy

def ppo_losses(new_logp, old_logp, adv, new_value, ret, clip=0.2, weight=None):
    """Calculates two PPO losses for one batch of B steps.
    weight (optional, no gradient) multiplies each step's policy term: the lag fix's correction for steps that
    were played by older weights than old_logp's"""
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    if weight is not None:
        surrogate = weight * surrogate
    policy_loss = -surrogate.mean()
    value_loss = (new_value - ret).square().mean()
    return policy_loss, value_loss

def normalise(adv):
    """Normalizes advantages to have a mean of 0 and var of 1"""
    return (adv - adv.mean()) / (adv.std() + 1e-8)

def scale_adv(adv, scale=None):
    """Scales grades used for the push in one minibatch"""
    if scale is None:
        return normalise(adv)

    return (adv / scale)
 
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
 
def page_logp(policy, obs_all, data_all, chunk=512):
    """Log probability of every stored move under the policy's current weights, without gradients"""
    out = []
    with torch.no_grad():
        for s in range(0, len(data_all["actions"]), chunk):
            obs = {k: v[s:s + chunk] for k, v in obs_all.items()}
            logits, _ = policy(obs)
            out.append(pick(masked_log_probs(logits, data_all["masks"][s:s + chunk]), data_all["actions"][s:s + chunk]))
    return torch.cat(out)

def update(policy, optimizer, buffer, obs_to_torch, epochs=4, minibatch=512, clip=0.2,
           vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, rng=None, adv_scale=None, lag="off", w_cap=2.0):
    """Learn from one full buffer page, doing 'ephocs' passes, each split into random groups of `minibatch` steps.

    lag is for pages played by older weights than the policy's (overlap):
    - "off": plain PPO, the clip is measured from the weights that played the page.
    - "log": plain PPO, but also logs how far apart the playing weights and the current ones are (lag_kl, lag_clip).
    - "fix": decoupled PPO (Hilton et al. 2021). The clip is measured from the current weights (the 'proximal'
      policy), so the full clip room is there again, and each step is weighted by p_current / p_played
      (capped at w_cap) to correct for the page having been played by the older weights."""
    if lag not in ("off", "log", "fix"):
        raise ValueError(f"lag must be off, log or fix, not {lag!r}")
    rng = rng or np.random.default_rng()
    device = next(policy.parameters()).device
    total = buffer.T * buffer.N
    log = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [], "push": []}
    obs_all, data_all = buffer.get(np.arange(total))
    obs_all = obs_to_torch(obs_all, device)
    data_all = to_tensors(data_all, device)

    played = data_all["logp"]
    centre, weight, extra = played, None, {}
    if lag != "off":
        prox = page_logp(policy, obs_all, data_all)
        lag_kl, lag_clip = health(prox, played, clip)
        extra = {"lag_kl": lag_kl, "lag_clip": lag_clip}
        log["clip_frac_played"] = []
        if lag == "fix":
            raw = torch.exp(prox - played)
            weight = raw.clamp(max=w_cap)
            extra["w_capped"] = float((raw > w_cap).float().mean().item())
            centre = prox

    for epoch in range(epochs):
        order = rng.permutation(total)
        for start in range(0, total, minibatch):
            idx = torch.as_tensor(order[start:start + minibatch], device=device)
            obs = {k: v[idx] for k, v in obs_all.items()}
            d = {k: v[idx] for k, v in data_all.items()}  
 
            logits, new_value = policy(obs)
            log_p = masked_log_probs(logits, d["masks"])
            new_logp = pick(log_p, d["actions"])
            ent = entropy(log_p).mean()
            adv = scale_adv(d["adv"], adv_scale)
            pl, vl = ppo_losses(new_logp, centre[idx], adv, new_value, d["ret"], clip,
                                None if weight is None else weight[idx])
            loss = pl + vf_coef * vl - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
 
            kl, frac = health(new_logp, centre[idx], clip)
            if lag != "off":
                log["clip_frac_played"].append(health(new_logp, d["logp"], clip)[1])
            log["policy_loss"].append(pl.item())
            log["value_loss"].append(vl.item())
            log["entropy"].append(ent.item())
            log["approx_kl"].append(kl)
            log["clip_frac"].append(frac)
            log["push"].append(adv.abs().mean().item())
 
    return {**{k: float(np.mean(v)) for k, v in log.items()}, **extra}

def linear_lr(r, rounds, lr_max):
    """Learning rate for round r of a run that has played 'r' rounds"""
    return lr_max * (1.0 - (r / rounds))