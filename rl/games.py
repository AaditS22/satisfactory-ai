import multiprocessing as mp
import traceback

import numpy as np

def _reset(envs, seeds):
    """Start every env with its seed"""
    obs = [env.reset(seed=s)[0] for env, s in zip(envs, seeds)]
    return obs, [env.action_masks() for env in envs]

def _play(envs, actions):
    """Step every env once"""
    obs, masks, rewards, dones, infos = [], [], [], [], []
    for env, a in zip(envs, actions):
        o, r, term, trunc, info = env.step(a)
        done = term or trunc
        if done:
            info = dict(info)
            info["score"] = r
            o, _ = env.reset()
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

    def step(self, actions):
        return _pack(*_play(self.envs, actions))

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
                out = _play(envs, data)
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

    def step(self, actions):
        actions = np.asarray(actions)
        return _pack(*self._ask("step", [actions[idx] for idx in self.split]))

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