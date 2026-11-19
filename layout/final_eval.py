import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from layout.anneal import ITERS
from layout.best_of import EvalTasks
from layout.design import MODES, SEED, SLOTS, WORKERS, check_fits, design, load_models, play
from layout.evaluate import EVAL_PATH
from layout.runlog import kind
from layout.tasks import random_task, targets, valid_rates
from layout.testset import STAGES, run_one
from rl.games import ProcGames
from solver.gamedata import items
from solver.graph import build_graph
from solver.lp_solve import solve_lp

HELD_SEED = 70000
TRIES = 50  # task draws per target before giving up on it
OUT = Path("data/final_eval.json")

def held_targets():
    """Every item the LP can make at some valid rate that no training stage uses"""
    trained = {t for stage in STAGES for t in stage}
    return [t for t in targets() if t not in trained and valid_rates(solve_lp(t, 1, None)[0])]

def trained_parts():
    """Every item made inside some factory of the test set (drawn like the training tasks)"""
    parts = set()
    for row in json.loads(Path(EVAL_PATH).read_text()):
        t = row["task"]
        for e in build_graph(t["target"], t["rate"], t["alternates"])["edges"]:
            if e["from"] != "RAW":
                parts.add(e["item"])
    return parts

def held_tasks(per_target):
    """Up to per_target random tasks for every held-out target, drawn like the test set, that the models can play"""
    res = []
    for t, target in enumerate(held_targets()):
        found = 0
        for k in range(TRIES):
            seed = HELD_SEED + 1000 * t + k
            task = random_task(np.random.default_rng(seed), target)
            try:
                check_fits(task)
            except ValueError:
                continue
            res.append((target, seed, task))
            found += 1
            if found == per_target:
                break
    return res

def name(target):
    """In-game name of an item"""
    return items[target]["name"]

def se(d):
    """Standard error of a mean"""
    return float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")

def summary(rows):
    """Prints how each mode does against the baselines and against "all", on the tasks Baseline A can solve"""
    solv = [r for r in rows if r["A"]["score"] > 0]
    A = np.array([r["A"]["score"] for r in solv])
    B = np.array([r["B"]["score"] for r in solv])
    print(f"{len(rows)} tasks over {len(set(r['target'] for r in rows))} held-out targets, "
          f"{len(solv)} that Baseline A can solve (only those are counted)\n")

    print("method        mean   vs A            vs B            beats A  >= B  broken  same pick  loss vs all  time/task  games")
    print(f"Baseline A    {A.mean():.3f}{'':79s}{np.mean([r['A']['time'] for r in solv]):7.2f} s")
    print(f"Baseline B    {B.mean():.3f}{'':79s}{np.mean([r['B']['time'] for r in solv]):7.2f} s")
    best_all = np.array([r["all"]["score"] for r in solv])
    for mode in MODES:
        s = np.array([r[mode]["score"] for r in solv])
        same = np.mean([r[mode]["stage"] == r["all"]["stage"] for r in solv])
        broken = sum(r[mode]["kind"] != "valid" for r in solv)
        print(f"RL {mode:9s}  {s.mean():.3f}  {(s - A).mean():+.3f} ({se(s - A):.3f})  {(s - B).mean():+.3f} ({se(s - B):.3f})  "
              f"{np.mean(s > A):6.0%}  {np.mean(s >= B):4.0%}  {broken:6d}  {same:8.0%}  {(best_all - s).mean():+11.4f}  "
              f"{np.mean([r[mode]['time'] for r in solv]):7.2f} s  {solv[0][mode]['games']:5d}")
    print("(vs A / vs B: mean paired difference (standard error). B time is one core per task, with several running at once)\n")

    print("target                         seen as part  tasks  A-solv     A      B    all  greedy  scout  stages picked by all")
    for target in sorted(set(r["target"] for r in rows), key=name):
        g = [r for r in rows if r["target"] == target]
        gs = [r for r in g if r["A"]["score"] > 0]
        means = "".join(f"{np.mean([r[m]['score'] for r in gs]) if gs else float('nan'):7.3f}"
                        for m in ["A", "B"] + MODES)
        picks = {}
        for r in g:
            picks[r["all"]["stage"]] = picks.get(r["all"]["stage"], 0) + 1
        picked = " ".join(f"{s}:{n}" for s, n in sorted(picks.items()))
        print(f"{name(target):30s} {'yes' if g[0]['part'] else 'no':>12s}  {len(g):5d}  {len(gs):6d}{means}   {picked}")

def main():
    parser = argparse.ArgumentParser(description="Final verification: the baselines and every design mode on tasks for targets no stage trained on")
    parser.add_argument("--per-target", type=int, default=4, help="tasks per held-out target")
    parser.add_argument("--show", action="store_true", help="only print the summary of the saved file")
    args = parser.parse_args()

    if args.show:
        summary(json.loads(OUT.read_text()))
        return

    parts = trained_parts()
    tasks = held_tasks(args.per_target)
    print(f"{len(tasks)} tasks over {len(set(t[0] for t in tasks))} held-out targets, running Baselines A and B")
    with Pool(WORKERS) as pool:
        rows = pool.map(run_one, [(0, target, seed, task, ITERS) for target, seed, task in tasks])
    for r in rows:
        r["part"] = r["target"] in parts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    n = len(rows)
    with ProcGames(EvalTasks([r["task"] for r in rows]), SLOTS, WORKERS) as games:
        play(games, models[0], 1, task=0, n_tasks=n)
        for i, r in enumerate(rows):
            for mode in MODES:
                gen = torch.Generator(device=device).manual_seed(SEED)
                t0 = time.time()
                agent, stage, looks, games_played = design(games, models, mode, gen, task=i, n_tasks=n)
                r[mode] = {"score": agent["score"], "kind": kind(agent), "stage": stage, "looks": [x["score"] for x in looks],
                           "games": games_played, "time": time.time() - t0, "plan": agent["plan"]}
            OUT.write_text(json.dumps(rows[:i + 1], indent=1))
            print(f"\r{i + 1}/{n} tasks", end="", flush=True)
    print(f"\nsaved {n} tasks to {OUT}\n")
    summary(rows)

if __name__ == "__main__":
    main()