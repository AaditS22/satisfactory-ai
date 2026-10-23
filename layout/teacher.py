import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
 
import numpy as np
 
from layout.anneal import baseline_b
from layout.env import LayoutEnv
from layout import obs as O
from layout.gym_env import StageEnv, observe, pad_mask, snap_mask, S, POS_SHAPE, N_POS
from layout.tasks import FLOORS, GRID_MAX
from solver.graph import build_graph

TEACH_SEED = 0 # safe to start at 0, eval uses 5000+
GRID_SHAPE = (O.N_CH, FLOORS, GRID_MAX, GRID_MAX)
SMALL_KEYS = ["vec", "stage", "nodes", "node_mask", "edges", "edge_rate", "edge_mask"]
EVAL_PATH = Path("data/eval_set.json")

def teacher_task(stage, seed):
    """The random training task with this seed (same generator as PPO training)"""
    return StageEnv(stage).task(np.random.default_rng(seed))

def run_one(args):
    """Baseline B on one teacher task -> one record (alternates saved as a sorted list)"""
    stage, seed, iters = args
    task = teacher_task(stage, seed)
    res = baseline_b(task, iters=iters, seed=seed)
    t = dict(task)
    t["alternates"] = sorted(t["alternates"])
    return {"seed": seed, "task": t, "plan": res["plan"], "score": float(res["score"])}

def factory_key(task):
    g = build_graph(task["target"], task["rate"], alternates=frozenset(task.get("alternates") or ()))
    recipes = tuple(sorted((r, tuple(round(c, 4) for c in d["clocks"])) for r, d in g["nodes"].items()))
    return (task["target"], task["rate"], task["width"], task["depth"], task["floors"], task["side"], recipes)
 
def eval_keys(path=EVAL_PATH):
    return {factory_key(r["task"]) for r in json.loads(Path(path).read_text())}

def load_teacher(path, eval_path=EVAL_PATH):
    """Read teacher records back, only keeping working layouts"""
    banned = eval_keys(eval_path)
    keep = []
    dropped = 0
    for r in json.loads(Path(path).read_text()):
        if r["score"] <= 0:
            continue
        r["task"]["alternates"] = frozenset(r["task"]["alternates"])
        if factory_key(r["task"]) in banned:
            dropped += 1
            continue
        keep.append(r)
    print(f"teacher data: {len(keep)} records kept, {dropped} dropped because they are eval tasks")
    return keep

def steps(task, plan):
    """Replay a 1 m plan move by move, returing the (obs, mask, action) the network would face"""
    env = LayoutEnv()
    env.reset(task)
    out = []
    for n, rot, f, x, y in plan:
        mask = np.zeros(N_POS, dtype=bool)
        nm = env.n_mask()
        mask[:len(nm)] = nm
        out.append((observe(env, 0, 0), mask, int(n)))

        mask = pad_mask(snap_mask(env.pos_mask(n))).reshape(-1)
        a = int(np.ravel_multi_index((rot, f, x // S, y // S), POS_SHAPE))
        out.append((observe(env, 1, n), mask, a))

        env.step(n, rot, f, x, y)
    return out

def split_rows(rows, val_frac=0.1):
    """Split teacher records into test and train sets"""
    rows = sorted(rows, key=lambda r: r["seed"])
    n_val = max(1, int(round(len(rows) * val_frac)))
    return rows[:-n_val], rows[-n_val:]
 
def build(rows):
    """Turn teacher records into one dict of arrays"""
    grids, masks, small, actions, scores = [], [], {k: [] for k in SMALL_KEYS}, [], []
    for r in rows:
        for obs, mask, a in steps(r["task"], r["plan"]):
            grids.append(np.packbits(obs["grid"].reshape(-1)))
            masks.append(np.packbits(mask))
            for k in SMALL_KEYS:
                small[k].append(obs[k])
            actions.append(a)
            scores.append(r["score"])
    out = {"grid_bits": np.stack(grids), "mask_bits": np.stack(masks),
           "action": np.array(actions, dtype=np.int64), "score": np.array(scores, dtype=np.float32)}
    for k in SMALL_KEYS:
        out[k] = np.stack(small[k])
    return out
 
def batch(samples, idx):
    """Unpacks rows at positions idx"""
    B = len(idx)
    grid = np.unpackbits(samples["grid_bits"][idx], axis=1).reshape(B, *GRID_SHAPE)
    masks = np.unpackbits(samples["mask_bits"][idx], axis=1, count=N_POS).astype(bool)
    obs = {"grid": grid}
    for k in SMALL_KEYS:
        obs[k] = samples[k][idx]
    return obs, masks, samples["action"][idx], samples["score"][idx]
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--tasks", type=int, default=4000)
    p.add_argument("--iters", type=int, default=1000, help="annealing tries per task")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=None, help="default: data/teacher_stage<N>.json")
    args = p.parse_args()
 
    out = Path(args.out or f"data/teacher_stage{args.stage}.json")
    jobs = [(args.stage, TEACH_SEED + i, args.iters) for i in range(args.tasks)]
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"\r{k + 1}/{len(jobs)} tasks, {time.time() - t0:.0f} s", end="", flush=True)
    print()
    rows.sort(key=lambda r: r["seed"])
    out.write_text(json.dumps(rows))
 
    s = np.array([r["score"] for r in rows])
    n_steps = sum(2 * len(r["plan"]) for r in rows if r["score"] > 0)
    print(f"saved {out}: {len(rows)} tasks, {(s > 0).sum()} working layouts, {n_steps} training steps, "
          f"mean working score {s[s > 0].mean():.3f}")
 
if __name__ == "__main__":
    main()