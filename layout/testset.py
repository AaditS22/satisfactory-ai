import argparse
import json
import time
from pathlib import Path

import numpy as np

from layout.anneal import anneal
from layout.baseline import baseline_a
from layout.tasks import random_task

STAGES = [
    ["Desc_IronPlate_C", "Desc_IronRod_C", "Desc_Wire_C"],
    ["Desc_IronPlateReinforced_C", "Desc_Rotor_C"],
    ["Desc_ModularFrame_C", "Desc_SteelPlateReinforced_C"],
    ["Desc_Motor_C", "Desc_ModularFrameHeavy_C"],
    ["Desc_Computer_C"],
]
EVAL_SEED = 50000
OUT = Path("data/eval_set.json")

def eval_tasks(per_target):
    """Generates random tasks for every target"""
    res = []
    t = 0
    for stage, targets in enumerate(STAGES, start=1):
        for target in targets:
            for k in range(per_target):
                seed = EVAL_SEED + 1000 * t + k
                res.append((stage, target, seed, random_task(np.random.default_rng(seed), target)))
            t += 1
    return res

def run_one(args):
    """Baseline A and Baseline B on one task."""
    stage, target, seed, task, iters = args
    t0 = time.perf_counter()
    a = baseline_a(task)
    t1 = time.perf_counter()
    b = anneal(task, a["plan"], iters, np.random.default_rng(seed))
    t2 = time.perf_counter()

    t = dict(task)
    t["alternates"] = sorted(t["alternates"]) if t.get("alternates") is not None else None
    return {"stage": stage, "target": target, "seed": seed, "task": t,
            "A": {"plan": a["plan"], "score": a["score"], "time": t1 - t0},
            "B": {"plan": b["plan"], "score": b["score"], "time": t2 - t1, "iters": iters}}

def summary(rows):
    print("stage  target                        tasks  A valid  B valid  A median  B median  gain   B time")
    for stage, targets in enumerate(STAGES, start=1):
        for target in targets:
            g = [r for r in rows if r["target"] == target]
            if not g:
                continue
            A = np.array([r["A"]["score"] for r in g])
            B = np.array([r["B"]["score"] for r in g])
            both = (A > 0) & (B > 0)
            a_med = f"{np.median(A[A > 0]):.3f}" if (A > 0).any() else "  -  "
            b_med = f"{np.median(B[B > 0]):.3f}" if (B > 0).any() else "  -  "
            gain = f"{(B[both] - A[both]).mean():+.3f}" if both.any() else "  -  "
            tb = np.median([r["B"]["time"] for r in g])
            name = target.replace("Desc_", "").replace("_C", "")
            print(f"{stage:5d}  {name:28s}  {len(g):5d}  {(A > 0).sum():7d}  {(B > 0).sum():7d}  "
                  f"{a_med:>8s}  {b_med:>8s}  {gain:>5s}  {tb:5.1f} s")

def main():
    parser = argparse.ArgumentParser(description="Run Baselines A and B on the test targets and save the results.")
    parser.add_argument("--per-target", type=int, default=10, help="tasks per target")
    parser.add_argument("--iters", type=int, default=1000, help="annealing tries for Baseline B")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="only print the summary of the saved file")
    args = parser.parse_args()

    if args.show:
        summary(json.loads(OUT.read_text()))
        return

    jobs = [(stage, target, seed, task, args.iters) for stage, target, seed, task in eval_tasks(args.per_target)]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            rows = pool.map(run_one, jobs)
    else:
        rows = []
        for k, job in enumerate(jobs):
            rows.append(run_one(job))
            print(f"\r{k + 1}/{len(jobs)} tasks", end="", flush=True)
        print()

    OUT.write_text(json.dumps(rows, indent=1))
    print(f"saved {len(rows)} tasks to {OUT}\n")
    summary(rows)

if __name__ == "__main__":
    main()