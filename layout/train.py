import argparse
import time

import numpy as np
import torch

from rl.buffer import RolloutBuffer
from rl.collect import collect, act, Background
from rl.games import ProcGames
from rl.ppo import update, linear_lr
from layout.gym_env import StageEnv, N_POS
from layout.policy import Policy, obs_to_torch
from layout.runlog import RunLog, round_stats, line
from layout.evaluate import load_eval, evaluate, TIE
from rl.checkpoint import save_checkpoint, load_checkpoint 

def gpu_gb(device):
    """Peak GPU memory used so far, in GB (0 on CPU)"""
    return torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0

def choose_greedy(policy, obs_list, masks):
    """Chooses the most likely action without randomness"""
    actions, _, _ = act(policy, obs_list, masks, obs_to_torch, greedy=True)
    return actions

def train(stage=1, rounds=50, n_games=48, workers=24, T=32, lr=2.5e-4, epochs=4, minibatch=256, ent_coef=0.01, seed=0,
          device=None, run=None, eval_every=10, images_every=50, resume=False, solvable=False, init=None, overlap=False, verbose=True): 
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
    if resume and init:
        raise ValueError("use --resume (same run) or --init (new run), not both")
    
    run = run or time.strftime("%Y%m%d-%H%M%S")
    settings = {"stage": stage, "rounds": rounds, "ent_coef": ent_coef, "solvable": solvable, "overlap": overlap}

    start, best = 0, float("-inf") 
    if resume:
        last_round, extra = load_checkpoint(f"runs/{run}/last.pt", policy, optimizer, device)
        for k, v in settings.items():
            if k in extra and extra[k] != v:
                raise ValueError(f"run {run} was started with {k}={extra[k]}, not {v}; --resume needs the same "
                                f"settings (a new run from these weights: --init runs/{run}/last.pt)")
        start, best = last_round + 1, extra["best_eval"]
        print(f"resuming {run} at round {start} (best eval so far {best:.3f})")
    elif init:
        load_checkpoint(init, policy, device=device)
        print(f"new run {run} starting from the weights in {init}")

    log = RunLog(f"runs/{run}/log.csv", append=resume)
    eval_rows = load_eval(stage)
    eval_log = RunLog(f"runs/{run}/eval.csv", group="eval", append=resume, share=log)

    history = []
    with ProcGames(StageEnv(stage, solvable), n_games, workers) as games:
        obs, masks = games.reset(seed * 1000 + start * n_games)
        if overlap:
            bg = Background(policy, games, [buffer, RolloutBuffer(T, n_games, N_POS)], obs_to_torch, gen)
            bg.sync(policy)
            bg.start(obs, masks)
        for r in range(start, rounds): 
            lr_now = linear_lr(r, rounds, lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            t0 = time.time()

            if overlap:
                page, obs, masks, finished, play_s = bg.wait()
                bg.sync(policy)
                if r + 1 < rounds:
                    bg.start(obs, masks)
            else:
                obs, masks, finished = collect(games, obs, masks, policy, buffer, obs_to_torch, gen)
                page, play_s = buffer, time.time() - t0

            t1 = time.time()
            stats = update(policy, optimizer, page, obs_to_torch, epochs, minibatch, ent_coef=ent_coef, rng=rng)
            t2 = time.time()
            row = {"round": r, **round_stats(finished), **stats, "lr": lr_now,
                   "collect_s": t1 - t0, "play_s": play_s, "update_s": t2 - t1, "gpu_gb": gpu_gb(device)}
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
 
                new_best = e["solv_score"] > best
                if new_best:
                    best = e["solv_score"]
                    save_checkpoint(f"runs/{run}/best.pt", policy, optimizer, r, {"best_eval": best, **settings})
                save_checkpoint(f"runs/{run}/last.pt", policy, optimizer, r, {"best_eval": best, **settings})
 
                if verbose:
                    print(f"      eval  score {e['score']:.3f}  (A {e['A']:.3f}, B {e['B']:.3f})  "
                          f"solvable {e['solv_score']:.3f} (A {e['solv_A']:.3f})  >= A-{TIE} on {e['gate_A']:.0%}  failed {e['failed']:.2f}  invalid {e['invalid']:.2f}  "
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
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--init", default=None, help="start a new run from this checkpoint's weights")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--solvable", action="store_true", help="train only on tasks Baseline A can fit on the site")
    p.add_argument("--overlap", action="store_true", help="overlap CPU and GPU processes")
    args = p.parse_args()
    t0 = time.time()
    _, history = train(stage=args.stage, rounds=args.rounds, n_games=args.games, workers=args.workers, T=args.T,
                minibatch=args.minibatch, seed=args.seed, device=args.device, run=args.run,
                eval_every=args.eval_every, images_every=args.images_every, resume=args.resume,
                lr=args.lr, init=args.init, ent_coef=args.ent_coef, solvable=args.solvable, overlap=args.overlap)
    print(f"last 10 rounds: mean score {np.nanmean(history[-10:]):.3f}   ({(time.time() - t0) / 60:.1f} min)")

if __name__ == "__main__":
    main()