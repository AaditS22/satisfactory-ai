import numpy as np
import gymnasium as gym
from gymnasium import spaces

import argparse
import time

from layout.env import LayoutEnv, N_MAX
from layout.tasks import random_task, one_belt, GRID_MAX, FLOORS
from layout import obs as O
from layout.score import score_episode
from layout.testset import STAGES
from layout.baseline import greedy
from layout.api import play  
from layout.connect import added_belt
from layout.score import BELT_REF

S = 2
COARSE = GRID_MAX // S
POS_SHAPE = (4, FLOORS, COARSE, COARSE)
N_POS = 4 * FLOORS * COARSE * COARSE

def offsets(s=S):
    """Cells inside one s x s square"""
    cells = [(dx, dy) for dx in range(s) for dy in range(s)]
    return sorted(cells, key=lambda c: (c[0] + c[1], c[0]))

def snap_mask(m, s=S):
    """Checks if any 1m cell inside is legal"""
    R, F, W, D = m.shape
    cw, cd = -(-W // s), -(-D // s)
    big = np.zeros((R, F, cw * s, cd * s), dtype=bool)
    big[:, :, :W, :D] = m 
    big = big.reshape(R, F, cw, s, cd, s)
    return big.any(axis=(3, 5))

def fine_cell(m, rot, f, cx, cy, s=S):
    """Finds the 1m cell where the coarse action puts the block"""
    W, D = m.shape[2], m.shape[3]
    for dx, dy in offsets(s):
        x, y = cx * s + dx, cy * s + dy
        if x < W and y < D and m[rot, f, x, y]:
            return (x, y)
    raise ValueError("no legal cell")

def pad_mask(m):
    out = np.zeros(POS_SHAPE, dtype=bool)
    _, F, W, D = m.shape
    out[:, :F, :W, :D] = m
    return out

def observe(env, stage, n):
    out = {"grid": O.grid_obs(env), "vec": O.vec_obs(env),
        "stage": np.array([stage, n / N_MAX], dtype=np.float32)}
    out.update(O.graph_obs(env))
    return out

def obs_space():
    return spaces.Dict({
        "grid": spaces.Box(0, 1, (O.N_CH, FLOORS, GRID_MAX, GRID_MAX), np.uint8),
        "vec": spaces.Box(0, O.TIGHT_CAP, (O.N_VEC,), np.float32),
        "stage": spaces.Box(0, 1, (2,), np.float32),
        "nodes": spaces.Box(0, 1, (O.MAX_NODES, O.N_NODE), np.float32),
        "node_mask": spaces.MultiBinary(O.MAX_NODES),
        "edges": spaces.Box(0, O.MAX_NODES - 1, (O.MAX_EDGES, 2), np.int32),
        "edge_rate": spaces.Box(0, 1, (O.MAX_EDGES,), np.float32),
        "edge_mask": spaces.MultiBinary(O.MAX_EDGES),
    })

class LayoutGym(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, task_fn=random_task):
        self.observation_space = obs_space()
        self.action_space = spaces.Discrete(N_POS)
        self.env = LayoutEnv()
        self.task_fn = task_fn
        self.task = None
        self.stage = 0
        self.n = 0
        self.mask = np.zeros(N_POS, dtype=bool)
        self.fine = None

    def _make_mask(self):
        """Creates a mask of legal actions"""
        mask = np.zeros(N_POS, dtype=bool)
        if self.env.done:
            return mask
        if self.stage == 0:
            nm = self.env.n_mask()
            mask[:len(nm)] = nm
        else:
            self.fine = self.env.pos_mask(self.n)
            mask[:] = pad_mask(snap_mask(self.fine)).reshape(-1)
        return mask

    def _info(self):
        return {"stage": self.stage, "failed": self.env.failed, "blocks": len(self.env.world.blocks)}

    def action_masks(self):
        """Checks which actions are currently legal"""
        return self.mask.copy()

    def reset(self, seed=None, options=None):
        """Starts a new episode"""
        super().reset(seed=seed)
        given = (options or {}).get("task")
        while True:
            self.task = given if given is not None else self.task_fn(self.np_random)
            self.env.reset(self.task)
            if not self.env.done:
                break
            if given is not None:
                raise ValueError("task has no legal first move")
        self.stage = 0
        self.n = 0
        self.wire_m = 0
        self.mask = self._make_mask()
        return observe(self.env, self.stage, self.n), self._info()

    def step(self, action):
        """Performs an action and receievs a reward if complete"""
        action = int(action)
        if self.env.done:
            raise ValueError("episode finished, call reset()")
        if not (0 <= action < N_POS and self.mask[action]):
            raise ValueError(f"illegal action {action} in stage {self.stage}")

        wire = 0.0
        if self.stage == 0:
            self.n = action
            self.stage = 1
        else:
            rot, f, cx, cy = (int(v) for v in np.unravel_index(action, POS_SHAPE))
            x, y = fine_cell(self.fine, rot, f, cx, cy)
            self.env.step(self.n, rot, f, x, y)
            metres = added_belt(self.env)
            self.wire_m += metres
            wire = metres / (len(self.env.graph["edges"]) * BELT_REF)
            self.stage = 0
            self.n = 0

        self.mask = self._make_mask()
        info = self._info()
        info["wire"] = wire
        reward = 0.0
        if self.env.done:
            reward, info["terms"] = score_episode(self.env)
            info["wire_total"] = self.wire_m

        return observe(self.env, self.stage, self.n), reward, self.env.done, False, info

def play_random(episodes, seed=0, check=False):
    """Plays random legal actions for many episodes"""
    gym_env = LayoutGym()
    rng = np.random.default_rng(seed)
    finished = failed = valid = steps = 0
    blocks = []
    scores = []
    for i in range(episodes):
        obs, info = gym_env.reset(seed=seed + i)
        terminated = False
        while not terminated:
            legal = np.flatnonzero(gym_env.action_masks())
            obs, reward, terminated, truncated, info = gym_env.step(int(legal[rng.integers(len(legal))]))
            steps += 1
            if check:
                assert gym_env.observation_space.contains(obs)
        failed += info["failed"]
        finished += not info["failed"]
        valid += reward > 0
        blocks.append(info["blocks"])
        scores.append(reward)
    return {"episodes": episodes, "finished": finished, "failed": failed, "valid": valid, "steps": steps,
        "mean_blocks": float(np.mean(blocks)), "max_blocks": int(max(blocks)),
        "mean_score": float(np.mean(scores)), "median_score": float(np.median(scores))}

def shaping_weight(r, rounds, w0):
    """Find the current reward shaping weight, which goes down linearly to 0 at rounds / 2"""
    return w0 * max(0.0, 1 - (r / (rounds / 2)))

class Shaped:
    """Adds reward shaping by subtracting the placement cost from the reward"""

    def __init__(self, games, weight=0.0):
        self.games = games
        self.weight = weight

    def reset(self, seed=0):
        return self.games.reset(seed)

    def step(self, actions):
        obs, masks, rewards, dones, infos = self.games.step(actions)
        wire = np.array([info.get("wire", 0.0) for info in infos], dtype=np.float32)
        rewards = rewards - self.weight * wire
        return obs, masks, rewards, dones, infos

def a_fits(task):
    """True if Baseline A's placement fits every block on the site (fixed rotation, then free rotation)"""
    for free_rot in (False, True):
        env = play(task, greedy(task, free_rot))
        if not env.failed:
            return True
    return False

class StageEnv:
    """Performs make env for one stage from the curriculum/test set"""

    def __init__(self, stage, solvable=False):
        self.stage = stage
        self.targets = STAGES[stage - 1]
        self.solvable = solvable

    def __call__(self):
        return LayoutGym(task_fn=self.task)

    def task(self, rng):
        """Random task for one of this stage's targets (with solvable=True redraw until Baseline A fits it)"""
        while True:
            target = self.targets[rng.integers(len(self.targets))]
            task = random_task(rng, target)
            if one_belt(task) and (not self.solvable or a_fits(task)):
                return task

def main():
    parser = argparse.ArgumentParser(description="Random agent through the Gymnasium wrapper.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="check every observation against observation_space")
    args = parser.parse_args()
    t = time.perf_counter()
    res = play_random(args.episodes, args.seed, args.check)
    dt = time.perf_counter() - t
    print(f"{res['episodes']} episodes: {res['finished']} finished ({res['valid']} valid), "
        f"{res['failed']} ran out of room, {res['steps']} steps, blocks mean {res['mean_blocks']:.1f} "
        f"max {res['max_blocks']}, score mean {res['mean_score']:.3f} median {res['median_score']:.3f}, "
        f"{1000 * dt / res['episodes']:.0f} ms/episode")

if __name__ == "__main__":
    main()