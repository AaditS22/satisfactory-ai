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

def collect(envs, obs, policy, buffer, obs_to_torch, gen=None):
    """Plays buffer.T ticks in N games at once, writes every tick into the buffer, then finishes it"""
    buffer.reset()
    scores = []
    for t in range(buffer.T):
        masks = np.stack([env.action_masks() for env in envs])
        actions, logp, values = act(policy, obs, masks, obs_to_torch, gen)
        rewards = np.zeros(buffer.N, dtype=np.float32)
        dones = np.zeros(buffer.N, dtype=bool)
        next_obs = []

        for i in range(len(envs)):
            o, r, term, trunc, info = envs[i].step(actions[i])
            rewards[i] = r
            dones[i] = term or trunc
            if dones[i]:
                scores.append(r)
                o, _ = envs[i].reset()

            next_obs.append(o)

        buffer.add(stack(obs), masks, actions, logp, values, rewards, dones)
        obs = next_obs

    masks = np.stack([env.action_masks() for env in envs])
    _, _, last_value = act(policy, obs, masks, obs_to_torch, gen)
    buffer.finish(last_value)
    return obs, scores