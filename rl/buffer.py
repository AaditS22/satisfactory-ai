import numpy as np

def gae(rewards, values, dones, last_value, gamma=1.0, lam=0.95):
    """Finds the adv of every step walking backwards from the last step"""
    adv = np.zeros_like(values)
    next_adv = np.zeros(values.shape[1])
    next_value = last_value

    for t in range(values.shape[0] - 1, -1, -1):
        keep = 1.0 - dones[t]
        delta = (rewards[t] + gamma * next_value * keep) - values[t]
        next_adv = delta + gamma * lam * keep * next_adv
        adv[t] = next_adv
        next_value = values[t]

    return adv, adv + values

def loo_advantage(returns):
    """Finds the score of every game compared with its copies"""
    G, K = returns.shape
    if K < 2:
        raise ValueError("K must be at least 2.")
    
    row_sums = returns.sum(axis=1, keepdims=True)
    others_avg = (row_sums - returns) / (K - 1)
    out = (returns - others_avg).astype(np.float32)
    
    return out

def fix_broken(returns, broken, margin):
    """Makes broken games count slightly worse than the worst instead of completely different"""
    out = returns.astype(np.float64)

    for g in range(out.shape[0]):
        b = broken[g]
        if not b.any():
            continue

        if b.all():
            out[g] = out[g].mean()
        else:
            worst = out[g, ~b].min()
            out[g, b] = worst - margin
    return out
    
class RolloutBuffer:
    """Writes T steps of N games. Fill it row by row with add(), then finish(), then read steps with get()."""

    def __init__(self, T, N, n_actions):
        self.T, self.N = T, N
        self.obs = None
        self.masks = np.zeros((T, N, n_actions), dtype=bool)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.logp = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.t = 0

    def add(self, obs, masks, actions, logp, values, rewards, dones):
        """Store one row"""
        if self.t == self.T:
            raise ValueError("buffer is full")
        if self.obs is None:
            self.obs = {k: np.zeros((self.T,) + v.shape, dtype=v.dtype) for k, v in obs.items()}
        for k, v in obs.items():
            self.obs[k][self.t] = v
        self.masks[self.t] = masks
        self.actions[self.t] = actions
        self.logp[self.t] = logp
        self.values[self.t] = values
        self.rewards[self.t] = rewards
        self.dones[self.t] = dones
        self.t += 1

    def finish(self, last_value, gamma=1.0, lam=0.95):
        """Compute advantages and value targets once the buffer is full"""
        if self.t != self.T:
            raise ValueError(f"buffer has {self.t} of {self.T} rows")
        self.adv, self.ret = gae(self.rewards, self.values, self.dones, last_value, gamma, lam)

    def flat(self, a, idx):
        """Flattens rows of an array as one long list of steps"""
        return a.reshape((self.T * self.N,) + a.shape[2:])[idx]

    def get(self, idx):
        """Gets the steps from the list in idx"""
        obs = {k: self.flat(v, idx) for k, v in self.obs.items()}
        data = {"masks": self.flat(self.masks, idx), "actions": self.flat(self.actions, idx),
                "logp": self.flat(self.logp, idx), "values": self.flat(self.values, idx),
                "adv": self.flat(self.adv, idx), "ret": self.flat(self.ret, idx)}
        return obs, data

    def reset(self):
        self.t = 0

class EpisodeBuffer:
    """Steps of whole games, where every task was repeated 'copies' times"""

    def __init__(self, copies, margin=0.1):
        self.copies, self.margin = copies, margin
        self.reset()

    def reset(self):
        self.open = {}
        self.games = []
        self.T, self.N = 0, 1

    def add(self, slot, obs, mask, action, logp, value):
        """Store one step of the game this slot is playing"""
        self.open.setdefault(slot, []).append((obs, mask, action, logp, value))

    def end_game(self, slot, job, score, broken):
        """Close the game this slot was playing"""
        self.games.append((job, self.open.pop(slot), score, broken))

    def finish(self):
        """Grade every game against the other copies of its task, then lay all steps out as one list"""
        if self.open:
            raise ValueError(f"{len(self.open)} games are still being played")
        rows = {}
        for g, game in enumerate(self.games):
            rows.setdefault(game[0], []).append(g)
        if not rows or any(len(r) != self.copies for r in rows.values()):
            raise ValueError(f"every task needs exactly {self.copies} finished copies")

        table = np.array(list(rows.values()))
        scores = np.array([game[2] for game in self.games], dtype=np.float64)
        broken = np.array([game[3] for game in self.games], dtype=bool)
        fixed = fix_broken(scores[table], broken[table], self.margin)
        grades = loo_advantage(fixed)
        game_adv = np.zeros(len(self.games), dtype=np.float32)
        game_adv[table] = grades

        steps = [(step, g) for g, game in enumerate(self.games) for step in game[1]]
        game_of = np.array([g for _, g in steps])
        self.obs = {k: np.stack([step[0][k] for step, _ in steps]) for k in steps[0][0][0]}
        self.masks = np.stack([step[1] for step, _ in steps])
        self.actions = np.array([step[2] for step, _ in steps], dtype=np.int64)
        self.logp = np.array([step[3] for step, _ in steps], dtype=np.float32)
        self.values = np.array([step[4] for step, _ in steps], dtype=np.float32)
        self.adv = game_adv[game_of]
        self.ret = scores[game_of].astype(np.float32)
        self.T = len(steps)
        self.stats = {"steps": self.T, "groups": len(table), "flat": float(np.mean(fixed.max(1) == fixed.min(1))),
                      "adv": float(np.abs(grades).mean())}
        self.games = []

    def get(self, idx):
        """Gets the steps in idx (same keys as RolloutBuffer.get)"""
        obs = {k: v[idx] for k, v in self.obs.items()}
        data = {"masks": self.masks[idx], "actions": self.actions[idx], "logp": self.logp[idx],
                "values": self.values[idx], "adv": self.adv[idx], "ret": self.ret[idx]}
        return obs, data