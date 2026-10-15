import math
import numpy as np
import copy
import argparse

import time
from layout.tasks import random_task
from layout.api import evaluate
from layout.baseline import baseline_a

from layout.view import draw
from multiprocessing import Pool

T0 = 0.02
T1 = 0.0005
STEP = 4
ITERS = 1000
KINDS = ["shift", "rotate", "swap", "resplit"]

def temperature(i, iters, t0=T0, t1=T1):
    """Calculates the temperature at the current stage"""
    if iters <= 1:
        return t0

    return t0 * (t1/t0) ** (i / (iters - 1))


def accept(old, new, t, u):
    """Determines if a new random change should be accepted, based on benefit and temperature"""
    if new >= old:
        return True

    return u < math.exp((new - old) / t)

def copy_plan(plan):
    """Returns a deep copy of the plan"""
    return copy.deepcopy(plan)

def shift(plan, i, dx, dy):
    """Returns a copy of the plan with block i shifted by dx and dy"""
    p = copy_plan(plan)

    p[i][3] += dx
    p[i][4] += dy
    return p

def rotate(plan, i, k):
    """Returns a copy of the plan with block i rotated by k turns"""
    p = copy_plan(plan)

    p[i][1] = (p[i][1] + k) % 4
    return p

def swap(plan, i, j):
    """Returns a copy of the plan with block i and block j swapping places"""
    p = copy_plan(plan)

    p[i][2], p[j][2] = p[j][2], p[i][2]
    p[i][3], p[j][3] = p[j][3], p[i][3]
    p[i][4], p[j][4] = p[j][4], p[i][4]
    return p

def resplit(plan, a, b):
    """Move one machine from block a to block b, assuming both make the same recipe"""
    p = copy_plan(plan)

    p[a][0] -= 1
    p[b][0] += 1

    if p[a][0] == 0:
        del p[a]

    return p

def neighbour(plan, recipes, rng):
    """Performs one random change to the plan"""
    if len(plan) == 0:
        return None

    kind = KINDS[int(rng.integers(len(KINDS)))]
    i = int(rng.integers(len(plan)))

    if kind == "shift":
        dx, dy = int(rng.integers(-STEP, STEP + 1)), int(rng.integers(-STEP, STEP + 1))
        while (dx, dy) == (0, 0):
            dx, dy = int(rng.integers(-STEP, STEP + 1)), int(rng.integers(-STEP, STEP + 1))

        return shift(plan, i, dx, dy)

    elif kind == "rotate":
        k = int(rng.integers(1, 4))
        return rotate(plan, i, k)

    elif kind == "swap":
        if len(plan) == 1:
            return None

        j = int(rng.integers(len(plan)))
        while j == i:
            j = int(rng.integers(len(plan)))

        return swap(plan, i, j)

    else:
        valid = []

        for k in range(len(recipes) - 1):
            if recipes[k] == recipes[k+1]:
                valid.append(k)

        if len(valid) == 0:
            return None

        k = valid[int(rng.integers(len(valid)))]

        if rng.random() < 0.5:
            return resplit(plan, k, k + 1)
        return resplit(plan, k + 1, k)
        
def anneal(task, plan, iters, rng):
    """Conducts simulated annealing on the plan based on the task"""
    cur = evaluate(task, plan)
    cur["plan"] = copy_plan(plan)
    best = cur

    for i in range(iters):
        t = temperature(i, iters)

        recipes = []
        for block in cur["env"].world.blocks:
            recipes.append(block["block"]["recipe"])

        new_plan = neighbour(cur["plan"], recipes, rng)
        if new_plan is None:
            continue

        try:
            new = evaluate(task, new_plan)
            if accept(cur["score"], new["score"], t, rng.random()):
                new["plan"] = new_plan
                cur = new

            if cur["score"] > best["score"]:
                best = cur

        except ValueError:
            continue

    return best

def baseline_b(task, iters=ITERS, seed=0):
    """Conducts the final baseline B and returns the best found plan + score"""
    start = baseline_a(task)
    rng = np.random.default_rng(seed)
    return anneal(task, start["plan"], iters, rng)

def compare_one(args):
    """Compares one run of baseline A vs B"""
    seed, iters = args
    task = random_task(np.random.default_rng(seed))

    t0 = time.perf_counter()
    a = baseline_a(task)
    t1 = time.perf_counter()
    b = anneal(task, a["plan"], iters, np.random.default_rng(seed))
    t2 = time.perf_counter()

    return {"seed": seed, "a": a["score"], "b": b["score"], "ta": t1 - t0, "tb": t2 - t1,
            "blocks_a": len(a["plan"]), "blocks_b": len(b["plan"])}

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline A and Baseline B on random tasks.")
    parser.add_argument("--tasks", type=int, default=0, help="run this many tasks (seeds 0..N-1) and print a summary")
    parser.add_argument("--iters", type=int, default=ITERS, help="annealing tries per task")
    parser.add_argument("--workers", type=int, default=1, help="processes to run in parallel")
    parser.add_argument("--seed", type=int, default=0, help="task seed for the picture")
    parser.add_argument("--out", default="anneal.png")
    args = parser.parse_args()

    if args.tasks <= 0:
        task = random_task(np.random.default_rng(args.seed))
        a = baseline_a(task)
        b = anneal(task, a["plan"], args.iters, np.random.default_rng(args.seed))
        print(f"seed {args.seed}: A {a['score']:.3f} -> B {b['score']:.3f}")
        print(f"A plan {a['plan']}")
        print(f"B plan {b['plan']}")
        draw(b["env"], args.out, title=f"Baseline B, seed {args.seed}, score {b['score']:.2f} (A {a['score']:.2f})",
             routes=b["routes"])
        print(f"saved {args.out}")
        return

    jobs = [(seed, args.iters) for seed in range(args.tasks)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(compare_one, jobs)
    else:
        res = []
        for k, job in enumerate(jobs):
            res.append(compare_one(job))
            print(f"\r{k + 1}/{len(jobs)} tasks", end="", flush=True)
        print()

    A = np.array([r["a"] for r in res])
    B = np.array([r["b"] for r in res])
    TA = np.array([r["ta"] for r in res])
    TB = np.array([r["tb"] for r in res])

    def tiers(s):
        return f"{(s > 0).sum()} valid, {((s < 0) & (s >= -1)).sum()} broken, {(s < -1).sum()} ran out of room"

    both = (A > 0) & (B > 0)
    if not both.any():
        print("no task was valid for both baselines, nothing to compare")
        return
    gain = B[both] - A[both]
    print(f"{args.tasks} tasks, {args.iters} tries each")
    print(f"A: {tiers(A)}")
    print(f"B: {tiers(B)}")
    print(f"fixed by B (A not valid, B valid): {((A <= 0) & (B > 0)).sum()}")
    print(f"valid median score: A {np.median(A[A > 0]):.3f}, B {np.median(B[B > 0]):.3f}")
    print(f"gain on tasks valid in both ({both.sum()}): mean {gain.mean():.3f}, median {np.median(gain):.3f}, "
          f"max {gain.max():.3f}; improved {(gain > 1e-9).sum()}, unchanged {(gain <= 1e-9).sum()}")
    print(f"blocks changed by resplit: {sum(r['blocks_a'] != r['blocks_b'] for r in res)} tasks")
    print(f"time per task: A median {1000 * np.median(TA):.0f} ms; B median {np.median(TB):.1f} s, max {TB.max():.1f} s")
    top = sorted(res, key=lambda r: r["b"] - r["a"], reverse=True)[:3]
    print("biggest gains (seed: A -> B): " + ", ".join(f"{r['seed']}: {r['a']:.3f} -> {r['b']:.3f}" for r in top))

if __name__ == "__main__":
    main()