import argparse
import time

import torch

from layout.api import evaluate
from layout.best_of import EvalTasks, play_jobs
from layout.evaluate import load_eval
from layout.policy import Policy, obs_to_torch
from layout.runlog import kind
from layout.view import draw_compare
from rl.checkpoint import load_checkpoint
from rl.collect import act
from rl.games import ProcGames
from solver.gamedata import items

def task_name(row):
    """Short description of an eval task, e.g. 'Motor at 7.5/min on a 96 x 96 m site'"""
    t = row["task"]
    return f"{items[row['target']]['name']} at {t['rate']}/min on a {t['width']} x {t['depth']} m site"

def list_tasks(rows):
    """Print the eval tasks of a stage with the baselines' scores"""
    print("task    target                         rate    site      A      B")
    for r in rows:
        t = r["task"]
        print(f"{r['seed']:<7} {items[r['target']]['name']:30s} {t['rate']:>5}/min {t['width']:>3}x{t['depth']:<4} "
              f"{r['A']['score']:6.3f} {r['B']['score']:6.3f}")

def try_model(policy, row, k=64, seed=0, workers=8, slots=16):
    """Plays one eval task once greedily and k times sampled; gives (greedy info, best sampled info)"""
    device = next(policy.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    def greedy(obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, greedy=True)[0]

    def sampled(obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, gen=gen)[0]

    with ProcGames(EvalTasks([row["task"]]), min(slots, k + 1), workers) as games:
        g = play_jobs(games, [0], greedy)[0]
        tries = play_jobs(games, list(range(1, k + 1)), sampled)
    best = max(tries.values(), key=lambda info: info["score"])
    return g, best

def main():
    p = argparse.ArgumentParser(description="play a trained model on one test task and compare it with the baselines")
    p.add_argument("model", help="a checkpoint, e.g. models/stage4.pt")
    p.add_argument("--stage", type=int, default=None, help="test stage (default: the model's own stage)")
    p.add_argument("--task", type=int, default=None, help="test task seed (leave out to list the stage's tasks)")
    p.add_argument("--k", type=int, default=64, help="sampled layouts to keep the best of")
    p.add_argument("--seed", type=int, default=0, help="seed for the sampled layouts")
    p.add_argument("--out", default=None, help="save a picture of A, B and the model's best layout here")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    _, extra = load_checkpoint(args.model, policy, device=device)
    policy.eval()
    stage = args.stage or extra.get("stage")
    if stage is None:
        p.error("this checkpoint does not say its stage, so pass --stage")
    rows = load_eval(stage)

    if args.task is None:
        list_tasks(rows)
        print("\npick one with --task SEED")
        return
    row = next((r for r in rows if r["seed"] == args.task), None)
    if row is None:
        p.error(f"task {args.task} is not a stage {stage} test task (leave out --task to list them)")

    t0 = time.time()
    g, best = try_model(policy, row, args.k, args.seed, args.workers)
    print(f"{task_name(row)}  ({time.time() - t0:.0f} s)\n")
    print(f"Baseline A (greedy heuristic)     {row['A']['score']:6.3f}")
    print(f"Baseline B (simulated annealing)  {row['B']['score']:6.3f}")
    print(f"model, greedy                     {g['score']:6.3f}  {kind(g)}")
    print(f"model, best of {args.k:<3}                {best['score']:6.3f}  {kind(best)}")

    if args.out:
        t = row["task"]
        draw_compare([("Greedy heuristic (Baseline A)", evaluate(t, row["A"]["plan"]), False),
                      ("Simulated annealing (Baseline B)", evaluate(t, row["B"]["plan"]), False),
                      (f"RL agent (best of {args.k})", evaluate(t, best["plan"]), True)],
                     args.out, task_name(row))
        print(f"\nsaved {args.out}")

if __name__ == "__main__":
    main()
