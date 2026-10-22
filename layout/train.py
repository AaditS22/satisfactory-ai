import argparse
import time

import numpy as np
import torch

from rl.buffer import RolloutBuffer
from rl.collect import collect, act
from rl.games import ProcGames
from rl.ppo import update
from layout.gym_env import StageEnv, N_POS
from layout.policy import Policy, obs_to_torch
from layout.runlog import RunLog, round_stats, line
from layout.evaluate import load_eval, evaluate
from rl.checkpoint import save_checkpoint, load_checkpoint 

def gpu_gb(device):
    """Peak GPU memory used so far, in GB (0 on CPU)"""
    return torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0

def choose_greedy(policy, obs_list, masks):
    """Chooses the most likely action without randomness"""
    actions, _, _ = act(policy, obs_list, masks, obs_to_torch, greedy=True)
    return actions

def train(stage=1, rounds=50, n_games=48, workers=24, T=32, lr=2.5e-4, epochs=4, minibatch=256, seed=0,
          device=None, run=None, eval_every=10, images_every=50, resume=False, verbose=True): 
    """Trains Policy on LayoutGym tasks from one curriculum stage with PPO"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(T, n_games, N_POS)

    if resume and run is None:
        raise ValueError("resume needs the run name (--run)")
    run = run or time.strftime("%Y%m%d-%H%M%S")

    start, best = 0, float("-inf") 
    if resume:
        last_round, extra = load_checkpoint(f"runs/{run}/last.pt", policy, optimizer, device)
        start, best = last_round + 1, extra["best_eval"]
        print(f"resuming {run} at round {start} (best eval so far {best:.3f})")

    log = RunLog(f"runs/{run}/log.csv", append=resume)
    eval_rows = load_eval(stage)
    eval_log = RunLog(f"runs/{run}/eval.csv", group="eval", append=resume, share=log)

    history = []
    with ProcGames(StageEnv(stage), n_games, workers) as games:
        obs, masks = games.reset(seed * 1000 + start * n_games)
        for r in range(start, rounds): 
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

            if r % eval_every == 0 or r == rounds - 1:
                t3 = time.time()
                k = 10 if (r % images_every == 0 or r == rounds - 1) else 0
                e, pngs = evaluate(policy, eval_rows, f"runs/{run}/eval/round_{r:04d}", choose_greedy, k=k)
                e["eval_s"] = time.time() - t3
                eval_log.write({"round": r, **e})
                for i, png in enumerate(pngs):
                    eval_log.image(f"best/{i}", png, r)
 
                new_best = e["score"] > best
                if new_best:
                    best = e["score"]
                    save_checkpoint(f"runs/{run}/best.pt", policy, optimizer, r, {"best_eval": best, "stage": stage})
                save_checkpoint(f"runs/{run}/last.pt", policy, optimizer, r, {"best_eval": best, "stage": stage})
 
                if verbose:
                    print(f"      eval  score {e['score']:.3f}  (A {e['A']:.3f}, B {e['B']:.3f})  "
                          f"beats A on {e['beats_A']:.0%}  failed {e['failed']:.2f}  invalid {e['invalid']:.2f}  "
                          f"{e['eval_s']:.1f} s" + ("  NEW BEST, saved" if new_best else ""), flush=True)  

    log.close()
    eval_log.close()

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
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--images-every", type=int, default=50)
    p.add_argument("--resume", action="store_true", help="continue runs/<run>/last.pt")
    args = p.parse_args()
    t0 = time.time()
    _, history = train(stage=args.stage, rounds=args.rounds, n_games=args.games, workers=args.workers, T=args.T,
                minibatch=args.minibatch, seed=args.seed, device=args.device, run=args.run,
                eval_every=args.eval_every, images_every=args.images_every, resume=args.resume)
    print(f"last 10 rounds: mean score {np.nanmean(history[-10:]):.3f}   ({(time.time() - t0) / 60:.1f} min)")

if __name__ == "__main__":
    main()