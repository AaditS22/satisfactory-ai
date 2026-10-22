import numpy as np
import torch
from torch import nn

import argparse
import time
 
from rl.buffer import RolloutBuffer
from rl.collect import collect
from rl.ppo import update
from rl.target_env import PickTarget
from rl.games import LocalGames 

import matplotlib.pyplot as plt

HIDDEN = 64

def obs_to_torch(obs, device="cpu"):
    """Turns a stacked observation dict into tensors on device"""
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}

class TargetNet(nn.Module):
    """A small network for PickTarget"""

    def __init__(self, n_actions=8):
        super().__init__()
        self.fc1 = nn.Linear(n_actions + 2, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.pi = nn.Linear(HIDDEN, n_actions)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, obs):
        x1 = torch.relu(self.fc1(torch.cat([obs["target"], obs["progress"]], dim=1)))
        x2 = torch.relu(self.fc2(x1))
        logits = self.pi(x2)
        v = self.v(x2).squeeze(1)

        return logits, v

def report(r, scores, stats):
    """Reports one line per round"""
    print(f"round {r:4d}  games {len(scores):4d}  score {np.mean(scores):.3f}  entropy {stats['entropy']:.3f}  "
          f"kl {stats['approx_kl']:.4f}  clip {stats['clip_frac']:.3f}  value_loss {stats['value_loss']:.4f}")

def train(rounds=100, n_games=16, T=32, lr=1e-3, epochs=4, minibatch=128, seed=0, verbose=True):
    """Trains TargetNet on PickTarget with PPO"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    games = LocalGames(PickTarget, n_games) 
    obs, masks = games.reset(seed * 1000)
 
    policy = TargetNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(T, n_games, 8)
 
    history = []
    for r in range(rounds):
        obs, masks, finished = collect(games, obs, masks, policy, buffer, obs_to_torch, gen)
        scores = [f["score"] for f in finished]
        stats = update(policy, optimizer, buffer, obs_to_torch, epochs, minibatch, rng=rng)
 
        history.append(float(np.mean(scores)))
        if verbose:
            report(r, scores, stats)
    return policy, history
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default=None)
    args = p.parse_args()
    t0 = time.time()
    _, history = train(rounds=args.rounds, seed=args.seed)
    print(f"last 10 rounds: mean score {np.mean(history[-10:]):.3f}   ({time.time() - t0:.1f} s)")
    if args.plot:
        plt.plot(history, label="PPO")
        plt.axhline(1.0, ls="--", c="gray", label="best possible")
        plt.axhline(0.25, ls=":", c="gray", label="random")
        plt.xlabel("round")
        plt.ylabel("mean score")
        plt.legend()
        plt.savefig(args.plot, dpi=100)
 
if __name__ == "__main__":
    main()