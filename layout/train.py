import argparse
import time

import numpy as np
import torch

from rl.buffer import RolloutBuffer
from rl.collect import collect
from rl.games import ProcGames
from rl.ppo import update
from layout.gym_env import StageEnv, N_POS
from layout.policy import Policy, obs_to_torch
from layout.runlog import RunLog, round_stats, line

def gpu_gb(device):
    """Peak GPU memory used so far, in GB (0 on CPU)"""
    return torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0

def train(stage=1, rounds=50, n_games=48, workers=24, T=32, lr=2.5e-4, epochs=4, minibatch=256, seed=0,
          device=None, run=None, verbose=True):
    """Trains Policy on LayoutGym tasks from one curriculum stage with PPO"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    run = run or time.strftime("%Y%m%d-%H%M%S")
    log = RunLog(f"runs/{run}/log.csv")   

    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(T, n_games, N_POS)

    history = []
    with ProcGames(StageEnv(stage), n_games, workers) as games:
        obs, masks = games.reset(seed * 1000)
        for r in range(rounds):
            t0 = time.time()
            obs, masks, finished = collect(games, obs, masks, policy, buffer, obs_to_torch, gen)
            t1 = time.time()
            stats = update(policy, optimizer, buffer, obs_to_torch, epochs, minibatch, rng=rng)
            t2 = time.time()
            row = {"round": r, **round_stats(finished), **stats,
                   "collect_s": t1 - t0, "update_s": t2 - t1, "gpu_gb": gpu_gb(device)}
            log.write(row)

            history.append(row["score"])
            if verbose:
                print(line(row), flush=True) 
    return policy, history

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--run", default=None)
    args = p.parse_args()
    t0 = time.time()
    _, history = train(stage=args.stage, rounds=args.rounds, n_games=args.games, workers=args.workers, T=args.T,
                       minibatch=args.minibatch, seed=args.seed, device=args.device)
    print(f"last 10 rounds: mean score {np.nanmean(history[-10:]):.3f}   ({(time.time() - t0) / 60:.1f} min)")

if __name__ == "__main__":
    main()