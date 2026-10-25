import multiprocessing as mp
import traceback

import numpy as np

def _reset(envs, seeds):
    """Start every env with its seed"""
    obs = [env.reset(seed=s)[0] for env, s in zip(envs, seeds)]
    return obs, [env.action_masks() for env in envs]

def _play(envs, actions, jobs=None):
    """Step every env once"""
    obs, masks, rewards, dones, infos = [], [], [], [], []
    for i, (env, a) in enumerate(zip(envs, actions)):
        o, r, done, info = None, 0.0, False, {}
        if a is not None:
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            if done:
                info = dict(info)
                info["score"] = r
                o = env.reset()[0] if jobs is None else None
        if o is None and jobs is not None and jobs[i] is not None:
            o, _ = env.reset(seed=jobs[i])
        obs.append(o)
        masks.append(env.action_masks())
        rewards.append(r)
        dones.append(done)
        infos.append(info)
    return obs, masks, rewards, dones, infos

def _pack(obs, masks, rewards, dones, infos):
    return obs, np.stack(masks), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

class LocalGames:
    """N games inside one program played one after another"""

    def __init__(self, make_env, n_games):
        self.n = n_games
        self.envs = [make_env() for _ in range(n_games)]

    def reset(self, seed=0):
        obs, masks = _reset(self.envs, [seed + i for i in range(self.n)])
        return obs, np.stack(masks)

    def step(self, actions, jobs=None):
        return _pack(*_play(self.envs, actions, jobs))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def _worker(conn, make_env, n):
    """Runs in its own process, owns n games"""
    envs = [make_env() for _ in range(n)]
    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            break
        try:
            if cmd == "reset":
                out = _reset(envs, data)
            else:
                out = _play(envs, *data)
            conn.send(("ok", out))
        except Exception:
            conn.send(("error", traceback.format_exc()))
    conn.close()

class ProcGames:
    """N games spread over worker processes"""

    def __init__(self, make_env, n_games, n_workers):
        ctx = mp.get_context("spawn")
        self.n = n_games
        self.split = np.array_split(np.arange(n_games), min(n_workers, n_games))
        self.conns, self.procs = [], []
        for idx in self.split:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, make_env, len(idx)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def _ask(self, cmd, per_worker):
        """Send every worker its part, wait for all answers, glue them back together in game order"""
        for conn, data in zip(self.conns, per_worker):
            conn.send((cmd, data))
        merged = None
        for conn in self.conns:
            status, out = conn.recv()
            if status == "error":
                raise RuntimeError("a game worker crashed:\n" + out)
            if merged is None:
                merged = [[] for _ in out]
            for k, part in enumerate(out):
                merged[k].extend(part)
        return merged

    def reset(self, seed=0):
        obs, masks = self._ask("reset", [[seed + int(i) for i in idx] for idx in self.split])
        return obs, np.stack(masks)

    def step(self, actions, jobs=None):
        actions = list(actions)
        parts = [([actions[i] for i in idx], None if jobs is None else [jobs[i] for i in idx]) for idx in self.split]
        return _pack(*self._ask("step", parts))

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

class JobQueue:
    """The jobs (reset seeds) of one page that no slot has started yet"""

    def __init__(self, jobs):
        self.jobs = list(jobs)

    def __len__(self):
        return len(self.jobs)

    def offer(self, playing):
        """Jobs to offer for one tick"""
        order = [i for i, p in enumerate(playing) if not p] + [i for i, p in enumerate(playing) if p]
        jobs = [None] * len(playing)
        for i, job in zip(order, self.jobs):
            jobs[i] = job
        return jobs

    def taken(self, jobs, playing, dones):
        """Removes the jobs that slots actually started this tick"""
        started = {}
        for i, job in enumerate(jobs):
            if job is not None and (not playing[i] or dones[i]):
                self.jobs.remove(job)
                started[i] = job
        return started