import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rl.buffer import RolloutBuffer, EpisodeBuffer
from rl.collect import collect, act, Background, collect_groups
from layout.runlog import RunLog, round_stats, line, run_dir, save_run_info, kind
from rl.games import ProcGames
from rl.ppo import update, linear_lr
from layout.gym_env import StageEnv, N_POS, Shaped, shaping_weight
from layout.policy import Policy, obs_to_torch
from layout.evaluate import load_eval, evaluate, TIE
from rl.checkpoint import save_checkpoint, load_checkpoint 

def gpu_gb(device):
    """Peak GPU memory used so far, in GB (0 on CPU)"""
    return torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0

def choose_greedy(policy, obs_list, masks):
    """Chooses the most likely action without randomness"""
    actions, _, _ = act(policy, obs_list, masks, obs_to_torch, greedy=True)
    return actions

def is_broken(info):
    """Checks if a game is broken or not"""
    return kind(info) != "valid"

def page_jobs(seed, r, groups, copies):
    """Reset seeds for round r's page"""
    return [1_000_000 * (seed + 1) + r * groups + g for g in range(groups) for _ in range(copies)]

def train(stage=1, rounds=50, n_games=48, workers=24, T=32, lr=2.5e-4, epochs=4, minibatch=256, ent_coef=0.01, seed=0,
          device=None, run=None, eval_every=10, images_every=50, resume=False, solvable=False, init=None, overlap=False, verbose=True,
          shaping=0.0, note="", copies=0, groups=40, broken_margin=0.1): 
    """Trains Policy on LayoutGym tasks from one curriculum stage with PPO"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    if copies and (overlap or shaping):
        raise ValueError("--copies can't be combined with --overlap or --shaping yet")
    buffer = EpisodeBuffer(copies, broken_margin) if copies else RolloutBuffer(T, n_games, N_POS)

    if resume and run is None:
        raise ValueError("resume needs the run name (--run)")
    if resume and init:
        raise ValueError("use --resume (same run) or --init (new run), not both")
    
    run = run or time.strftime("%Y%m%d-%H%M%S")

    folder = run_dir(stage, run)
    if resume and not Path(f"{folder}/last.pt").exists():
        raise ValueError(f"no run {run} in runs/stage{stage}/ to resume")
    if not resume and Path(f"{folder}/last.pt").exists():
        raise ValueError(f"{folder} already exists: pick a new --run name, or add --resume to continue it")
    
    settings = {"stage": stage, "rounds": rounds, "ent_coef": ent_coef, "solvable": solvable, "overlap": overlap,
            "shaping": shaping, "copies": copies, "groups": groups, "broken_margin": broken_margin}

    start, best = 0, float("-inf") 
    if resume:
        last_round, extra = load_checkpoint(f"{folder}/last.pt", policy, optimizer, device)
        for k, v in settings.items():
            if k in extra and extra[k] != v:
                raise ValueError(f"run {run} was started with {k}={extra[k]}, not {v}; --resume needs the same "
                                f"settings")
        start, best = last_round + 1, extra["best_eval"]
        print(f"resuming {run} at round {start} (best eval so far {best:.3f})")
    elif init:
        load_checkpoint(init, policy, device=device)
        print(f"new run {run} starting from the weights in {init}")

    log = RunLog(f"{folder}/log.csv", append=resume)
    eval_rows = load_eval(stage)
    eval_log = RunLog(f"{folder}/eval.csv", group="eval", append=resume, share=log)

    config = {"stage": stage, "rounds": rounds, "n_games": n_games, "workers": workers, "T": T, "lr": lr,
            "epochs": epochs, "minibatch": minibatch, "ent_coef": ent_coef, "seed": seed, "init": init,
            "solvable": solvable, "overlap": overlap, "shaping": shaping, "eval_every": eval_every,
            "images_every": images_every, "copies": copies, "groups": groups, "broken_margin": broken_margin}
    info = save_run_info(folder, note, config, resumed_at=start if resume else None)
    log.text("note", f"{note}  \n{info['command']}  \ncode {info['code']}" if not resume
             else f"resumed at round {start}: {note}")

    history = []
    with ProcGames(StageEnv(stage, solvable), n_games, workers) as raw:
        games = raw if copies else Shaped(raw)
        obs, masks = games.reset(seed * 1000 + start * n_games)
        if overlap:
            bg = Background(policy, games, [buffer, RolloutBuffer(T, n_games, N_POS)], obs_to_torch, gen)
            bg.sync(policy)
            games.weight = shaping_weight(start, rounds, shaping)
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
                    games.weight = shaping_weight(r + 1, rounds, shaping)
                    bg.start(obs, masks)
            elif copies:
                jobs = page_jobs(seed, r, groups, copies)
                finished, idle = collect_groups(games, policy, buffer, jobs, obs_to_torch, is_broken, gen)
                page, play_s = buffer, time.time() - t0
            else:
                games.weight = shaping_weight(r, rounds, shaping)
                obs, masks, finished = collect(games, obs, masks, policy, buffer, obs_to_torch, gen)
                page, play_s = buffer, time.time() - t0

            t1 = time.time()
            stats = update(policy, optimizer, page, obs_to_torch, epochs, minibatch, ent_coef=ent_coef, rng=rng)
            t2 = time.time()
            row = {"round": r, **round_stats(finished), **stats, "lr": lr_now,
                   "collect_s": t1 - t0, "play_s": play_s, "shaping": shaping_weight(r, rounds, shaping),
                    "update_s": t2 - t1, "gpu_gb": gpu_gb(device)}
            if copies:
                row.update(page.stats, idle=idle)
            log.write(row)

            history.append(row["score"])
            if verbose:
                print(line(row), flush=True) 

            if r % eval_every == 0 or r == rounds - 1:
                t3 = time.time()
                k = 10 if images_every and (r % images_every == 0 or r == rounds - 1) else 0
                e, pngs = evaluate(policy, eval_rows, f"{folder}/eval/round_{r:04d}", choose_greedy, k=k)
                e["eval_s"] = time.time() - t3
                eval_log.write({"round": r, **e})
                for i, png in enumerate(pngs):
                    eval_log.image(f"best/{i}", png, r)
 
                new_best = e["solv_score"] > best
                if new_best:
                    best = e["solv_score"]
                    save_checkpoint(f"{folder}/best.pt", policy, optimizer, r, {"best_eval": best, **settings})
                save_checkpoint(f"{folder}/last.pt", policy, optimizer, r, {"best_eval": best, **settings})
 
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
    p.add_argument("--images-every", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue runs/stage<N>/<run>/last.pt")
    p.add_argument("--note", default="", help="what this run changes (required for a new run)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--init", default=None, help="start a new run from this checkpoint's weights")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--solvable", action="store_true", help="train only on tasks Baseline A can fit on the site")
    p.add_argument("--overlap", action="store_true", help="overlap CPU and GPU processes")
    p.add_argument("--shaping", type=float, default=0.0)
    p.add_argument("--copies", type=int, default=0, help="play every task this many times per page, grade each game against its copies")
    p.add_argument("--groups", type=int, default=40, help="with --copies: tasks per page")
    p.add_argument("--broken-margin", type=float, default=0.1, help="with --copies: a broken game counts this much below its worst working copy")
    args = p.parse_args()

    if not args.resume and not args.note.strip():
        p.error("a new run needs --note")

    t0 = time.time()
    _, history = train(stage=args.stage, rounds=args.rounds, n_games=args.games, workers=args.workers, T=args.T,
                minibatch=args.minibatch, seed=args.seed, device=args.device, run=args.run,
                eval_every=args.eval_every, images_every=args.images_every, resume=args.resume,
                lr=args.lr, init=args.init, ent_coef=args.ent_coef, solvable=args.solvable, overlap=args.overlap,
                shaping=args.shaping, note=args.note, copies=args.copies, groups=args.groups,
                broken_margin=args.broken_margin)
    print(f"last 10 rounds: mean score {np.nanmean(history[-10:]):.3f}   ({(time.time() - t0) / 60:.1f} min)")

if __name__ == "__main__":
    main()