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