import numpy as np

class PickTarget:
    """Test environment with a known best possible score"""

    def __init__(self, n_actions=8, n_steps=3, n_legal=4):
        self.A, self.K, self.L = n_actions, n_steps, n_legal
        self.rng = np.random.default_rng()
        self.mask = np.zeros(self.A, dtype=bool)
        self.target = 0
        self.t = 0
        self.hits = 0
        self.done = True

    def _new_step(self):
        """Pick a new target and which other actions are legal"""
        self.target = int(self.rng.integers(self.A))
        others = np.delete(np.arange(self.A), self.target)
        self.mask = np.zeros(self.A, dtype=bool)
        self.mask[self.target] = True
        self.mask[self.rng.choice(others, self.L - 1, replace=False)] = True

    def _obs(self):
        target = np.zeros(self.A, dtype=np.float32)
        if not self.done:
            target[self.target] = 1.0
        progress = np.array([self.t / self.K, self.hits / self.K], dtype=np.float32)
        return {"target": target, "progress": progress}

    def action_masks(self):
        return self.mask.copy()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.hits = 0
        self.done = False
        self._new_step()
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if self.done:
            raise ValueError("episode finished, call reset()")
        if not (0 <= action < self.A and self.mask[action]):
            raise ValueError(f"illegal action {action}")
        if action == self.target:
            self.hits += 1
        self.t += 1
        reward = 0.0
        if self.t == self.K:
            self.done = True
            self.mask = np.zeros(self.A, dtype=bool)
            reward = self.hits / self.K
        else:
            self._new_step()
        return self._obs(), reward, self.done, False, {"hits": self.hits}