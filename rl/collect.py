import numpy as np
import torch

from rl.dist import masked_log_probs, sample, pick

def stack(obs_list):
    """Turns a list of observation dicts into one dict of arrays"""
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

def act(policy, obs_list, masks, obs_to_torch, gen=None):
    """Uses the network to sample actions to play for each game, with their probabilities and predicted scores"""
    device = next(policy.parameters()).device
    x = obs_to_torch(stack(obs_list), device)
    m = torch.as_tensor(masks, device=device)

    with torch.no_grad():
        logits, value = policy(x)
        log_p = masked_log_probs(logits, m)
        actions = sample(log_p, gen)
        logp = pick(log_p, actions)
        return actions.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

def collect(games, obs, masks, policy, buffer, obs_to_torch, gen=None):
    """Plays buffer.T ticks in N games at once, writes every tick into the buffer, then finishes it"""
    buffer.reset()
    finished = []
    for t in range(buffer.T):
        actions, logp, values = act(policy, obs, masks, obs_to_torch, gen)
        next_obs, next_masks, rewards, dones, infos = games.step(actions)

        buffer.add(stack(obs), masks, actions, logp, values, rewards, dones)

        for i in range(len(infos)):
            if dones[i]:
                finished.append(infos[i])

        obs, masks = next_obs, next_masks

    _, _, last_value = act(policy, obs, masks, obs_to_torch, gen)
    buffer.finish(last_value)
    return obs, masks, finished