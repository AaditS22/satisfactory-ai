import contextlib
import copy
import threading
import time

import numpy as np
import torch

from rl.dist import masked_log_probs, sample, pick
from rl.games import JobQueue

def stack(obs_list):
    """Turns a list of observation dicts into one dict of arrays"""
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

def act(policy, obs_list, masks, obs_to_torch, gen=None, greedy=False):
    """Uses the network to sample actions to play for each game, with their probabilities and predicted scores"""
    device = next(policy.parameters()).device
    x = obs_to_torch(stack(obs_list), device)
    m = torch.as_tensor(masks, device=device)

    with torch.no_grad():
        logits, value = policy(x)
        log_p = masked_log_probs(logits, m)
        actions = log_p.argmax(1) if greedy else sample(log_p, gen)
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

def collect_groups(games, policy, buffer, jobs, obs_to_torch, broken, gen=None):
    """Plays every job to the end and writes every game into an EpisodeBuffer, then finishes it"""
    buffer.reset()
    queue = JobQueue(jobs)
    n = games.n
    playing, current = [False] * n, [None] * n
    obs, masks = [None] * n, None
    finished, moves, ticks = [], 0, 0
    while len(queue) or any(playing):
        live = [i for i in range(n) if playing[i]]
        actions = [None] * n
        if live:
            a, logp, values = act(policy, [obs[i] for i in live], masks[live], obs_to_torch, gen)
            for k, i in enumerate(live):
                actions[i] = int(a[k])
                buffer.add(i, obs[i], masks[i], a[k], logp[k], values[k])

        offer = queue.offer(playing)
        obs, masks, rewards, dones, infos = games.step(actions, offer)

        for i in live:
            if dones[i]:
                buffer.end_game(i, current[i], infos[i]["score"], broken(infos[i]))
                finished.append(infos[i])
        for i, job in queue.taken(offer, playing, dones).items():
            current[i] = job
        playing = [o is not None for o in obs]
        moves += len(live)
        ticks += 1

    buffer.finish()
    return finished, 1.0 - moves / (n * ticks)

class Background:
    """Plays the next page in a second thread while the main thread learns from the current one"""

    def __init__(self, policy, games, buffers, obs_to_torch, gen=None):
        self.actor = copy.deepcopy(policy)
        self.games = games
        self.buffers = buffers
        self.obs_to_torch = obs_to_torch
        self.gen = gen
        device = next(policy.parameters()).device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.k = 0
        self.thread = None
        self.result = None
        self.error = None

    def sync(self, policy):
        """Copies the learner's weights into the actor. Only allowed while no page is being played."""
        if self.thread is not None:
            raise RuntimeError("a page is still being played")
        self.actor.load_state_dict(policy.state_dict())
        if self.stream is not None:
            torch.cuda.synchronize()

    def start(self, obs, masks):
        """Start playing the next page in the background"""
        if self.thread is not None:
            raise RuntimeError("a page is still being played")
        buffer = self.buffers[self.k]
        self.k = (self.k + 1) % len(self.buffers)
        self.result, self.error = None, None

        def job():
            try:
                t = time.perf_counter()
                where = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
                with where:
                    out = collect(self.games, obs, masks, self.actor, buffer, self.obs_to_torch, self.gen)
                self.result = (buffer, *out, time.perf_counter() - t)
            except BaseException as e:
                self.error = e

        self.thread = threading.Thread(target=job, daemon=True)
        self.thread.start()

    def wait(self):
        """Wait until the page is played, then return its results"""
        self.thread.join()
        self.thread = None
        if self.error is not None:
            raise self.error
        return self.result