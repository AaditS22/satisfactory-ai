import contextlib
import copy
import queue
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

def collect_groups_async(games, policy, buffer, jobs, obs_to_torch, broken, gen=None):
    """Same page as collect_groups, but every worker part moves on as soon as it answers.

    collect_groups makes all games wait for the slowest worker on every tick (a game's last step, which routes
    and scores, or a reset can take 100+ ms while a normal step takes ~3 ms). Here the network acts for whichever
    parts have answered, so one slow game only holds up the games in its own worker.
    Every game is still played by the same weights from start to end, so the page means the same thing; only
    the order of the random draws differs.
    Returns finished infos and the share of slot-steps sent with no move (idle)."""
    buffer.reset()
    queue = list(jobs)
    n, parts = games.n, games.split
    playing, current = [False] * n, [None] * n
    obs, masks = [None] * n, [None] * n
    offered = {}
    ready, busy = list(range(len(parts))), set()
    finished, moves, sent = [], 0, 0
    while True:
        send = [k for k in ready if any(playing[i] for i in parts[k]) or
                (queue and any(not playing[i] for i in parts[k]))]
        live = [i for k in send for i in parts[k] if playing[i]]
        actions = {}
        if live:
            a, logp, values = act(policy, [obs[i] for i in live], np.stack([masks[i] for i in live]), obs_to_torch, gen)
            for j, i in enumerate(live):
                actions[i] = int(a[j])
                buffer.add(i, obs[i], masks[i], a[j], logp[j], values[j])
        for i in [i for k in send for i in parts[k] if not playing[i]] + live:
            if not queue:
                break
            offered[i] = queue.pop(0)
        for k in send:
            idx = parts[k]
            games.send_part(k, [actions.get(int(i)) for i in idx], [offered.get(int(i)) for i in idx])
            busy.add(k)
            moves += sum(playing[i] for i in idx)
            sent += len(idx)
        ready = [k for k in ready if k not in send]
        if not busy:
            break

        unused = []
        for k, (o, m, _, dones, infos) in games.recv_parts():
            busy.discard(k)
            ready.append(k)
            for j, i in enumerate(parts[k]):
                i = int(i)
                was = playing[i]
                if was and dones[j]:
                    buffer.end_game(i, current[i], infos[j]["score"], broken(infos[j]))
                    finished.append(infos[j])
                job = offered.pop(i, None)
                if job is not None:
                    if not was or dones[j]:
                        current[i] = job
                    else:
                        unused.append(job)
                obs[i], masks[i] = o[j], m[j]
                playing[i] = o[j] is not None
        queue[:0] = unused

    buffer.finish()
    return finished, 1.0 - moves / max(1, sent)

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
        self._worker = None
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
        """Start playing the next page in the background; wait() gives (buffer, obs, masks, finished, seconds)"""
        self._launch(lambda buffer: collect(self.games, obs, masks, self.actor, buffer, self.obs_to_torch, self.gen))

    def start_groups(self, jobs, broken, use_async=True):
        """Start playing the next group page (every job to its end) in the background; wait() gives
        (buffer, finished, idle, seconds)"""
        play = collect_groups_async if use_async else collect_groups
        self._launch(lambda buffer: play(self.games, self.actor, buffer, jobs, self.obs_to_torch, broken, self.gen))

    def _launch(self, play):
        """Runs play(buffer) in the second thread, on the next buffer in turn"""
        if self.thread is not None:
            raise RuntimeError("a page is still being played")
        buffer = self.buffers[self.k]
        self.k = (self.k + 1) % len(self.buffers)
        self.result, self.error = None, None
        # One thread for every page, kept alive between pages. A new thread per page leaks ~0.3 MB of native
        # (C heap) memory each time it runs the policy on CUDA; over 2000 rounds that fragments the heap and
        # makes every numpy allocation in the process (eval included) several times slower.
        if self._worker is None:
            self._jobs, self._done = queue.Queue(), queue.Queue()
            self._worker = threading.Thread(target=_play_pages, args=(self._jobs, self._done, self.stream),
                                            daemon=True)
            self._worker.start()
        self._jobs.put((play, buffer))
        self.thread = self._worker

    def wait(self):
        """Wait until the page is played, then return its results"""
        self.result, self.error = self._done.get()
        self.thread = None
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        """Stop the second thread (it also stops on its own when the program ends)"""
        if getattr(self, "_worker", None) is not None:
            self._jobs.put(None)
            self._worker = None

    def __del__(self):
        self.close()

def _play_pages(jobs, done, stream):
    """The second thread: plays each page it is given and reports (result, error).
    It holds no reference to the Background, so a dropped Background (and its actor) can be freed."""
    while True:
        item = jobs.get()
        if item is None:
            return
        play, buffer = item
        try:
            t = time.perf_counter()
            where = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with where:
                out = play(buffer)
            done.put(((buffer, *out, time.perf_counter() - t), None))
        except BaseException as e:
            done.put((None, e))
        del item, play, buffer