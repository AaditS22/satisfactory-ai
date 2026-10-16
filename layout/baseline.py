import numpy as np

from layout.env import LayoutEnv
from layout.world import EMPTY, end_cells
from layout.connect import ends_for, FLOOR_COST
from layout.api import plan_of, evaluate
import argparse
import time
from layout.tasks import item_name, random_task

CHAIN = [(3, True), (3, False), (1, False)]
TIER_COST = 8.0
W_OUT = 0.3


def corridor_mask(env, w, d, g, axis=None):
    """Checks where w x d blocks can be placed and leave g cells around them"""
    F, X, Y = env.world.owner.shape

    big = np.zeros((F, X + 2 * g, Y + 2 * g), dtype=np.int32)
    big[:, g:g + X, g:g + Y] = env.world.owner != EMPTY

    if axis == 0:
        big[:, :g, :] = 1
        big[:, g + X:, :] = 1
    elif axis == 1:
        big[:, :, :g] = 1
        big[:, :, g + Y:] = 1

    S = np.zeros((F, X + 2 * g + 1, Y + 2 * g + 1), dtype=np.int32)
    S[:, 1:, 1:] = big.cumsum(axis=1).cumsum(axis=2)

    W, D = w + 2 * g, d + 2 * g
    count = S[:, W:, D:] - S[:, :-W, D:] - S[:, W:, :-D] + S[:, :-W, :-D]

    return count == 0

def wire_cost(env, blk, f, x, y):
    """Estimates a belt distance for input and output routing for a block at different position options"""
    ends = end_cells(blk)
    cost = np.zeros(len(f), dtype=np.float64)

    for k, inp in enumerate(blk["inputs"]):
        ex, ey = ends[k]
        producers = ends_for(env, inp["item"])[0] 

        best = np.full(len(f), np.inf)
        for pf, px, py in producers:
            dist = np.abs(x + ex - px) + np.abs(y + ey - py) + FLOOR_COST * np.abs(f - pf)
            best = np.minimum(best, dist)
        cost += best

    ox, oy = ends[-1]
    tx, ty = env.exit
    cost += W_OUT * (np.abs(x + ox - tx) + np.abs(y + oy - ty) + FLOOR_COST * f)

    return cost

def best_move(env, rot, axis):
    """Finds the cheapest placement of the queue head for one rotation"""
    for n in range(env.n_limit(), 0, -1):
        legal = env.pos_mask(n)[rot]
        if not legal.any():
            continue

        blk = env.block(n, rot)
        f, x, y = np.nonzero(legal)

        cost = wire_cost(env, blk, f, x, y)

        tier = np.full(len(f), len(CHAIN))
        for g, walls in CHAIN:
            cm = corridor_mask(env, blk["w"], blk["d"], g, axis if walls else None)
            tier = tier - cm[f, x, y]
        cost = cost + TIER_COST * tier

        i = int(np.argmin(cost))
        return (n, float(cost[i]), (n, rot, int(f[i]), int(x[i]), int(y[i])))

    return None

def greedy(task, free_rot=False):
    """Plays one whole episode with the greedy rules"""
    env = LayoutEnv()
    env.reset(task)

    side = task["side"]
    axis = side % 2

    while not env.done:
        if free_rot:
            rots = [0, 1, 2, 3]
        else:
            rots = [side]

        moves = []
        for rot in rots:
            m = best_move(env, rot, axis)
            if m is not None:
                moves.append(m)

        if not moves:
            for rot in range(4):
                m = best_move(env, rot, axis)
                if m is not None:
                    moves.append(m)

        n, cost, move = min(moves, key=lambda m: (-m[0], m[1]))
        env.step(*move)

    return plan_of(env)

def baseline_a(task):
    """First baseline for mode. Runs greedy with free and fixed rotation and picks the best"""
    best = None
    for free_rot in (False, True):
        plan = greedy(task, free_rot)
        res = evaluate(task, plan)
        res["plan"] = plan
        if best is None or res["score"] > best["score"]:
            best = res
    return best

def main():
    parser = argparse.ArgumentParser(description="Run Baseline A on random tasks.")
    parser.add_argument("--seed", type=int, default=0, help="task seed for the picture")
    parser.add_argument("--out", default="baseline.png")
    parser.add_argument("--tasks", type=int, default=0, help="if > 0: run this many tasks (seeds 0..N-1) and print a summary")
    parser.add_argument("--target", default=None, help="item for the picture")
    args = parser.parse_args()

    if args.tasks > 0:
        scores, times = [], []
        for seed in range(args.tasks):
            t0 = time.perf_counter()
            scores.append(baseline_a(random_task(np.random.default_rng(seed)))["score"])
            times.append(time.perf_counter() - t0)
        s = np.array(scores)
        valid = s > 0
        print(f"{args.tasks} tasks: {valid.sum()} valid, {((s < 0) & (s >= -1)).sum()} broken, {(s < -1).sum()} ran out of room")
        print(f"valid score median {np.median(s[valid]):.3f}, mean of all {s.mean():.3f}")
        print(f"time per task: median {1000 * np.median(times):.0f} ms, max {1000 * max(times):.0f} ms")
        return

    from layout.view import draw
    task = random_task(np.random.default_rng(args.seed), item_name(args.target))
    res = baseline_a(task) 
    print(f"seed {args.seed}: score {res['score']:.3f}, plan {res['plan']}")
    draw(res["env"], args.out, title=f"Baseline A, seed {args.seed}, score {res['score']:.2f}", routes=res["routes"])
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()