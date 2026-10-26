import argparse
import time

from layout.evaluate import load_eval, TIE
from layout.runlog import mean, kind
from layout.gym_env import LayoutGym
from layout.api import plan_of
from rl.games import ProcGames, JobQueue

import torch
from layout.policy import Policy, obs_to_torch
from rl.checkpoint import load_checkpoint
from rl.collect import act

import json
from pathlib import Path

class TaskGym(LayoutGym):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def reset(self, seed=None, options=None):
        if seed is not None and not (options or {}).get("task"):
            options = {"task": self.tasks[seed % len(self.tasks)]}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term or trunc:
            info["plan"] = plan_of(self.env)
        return obs, reward, term, trunc, info

class EvalTasks:
    """Picklable maker of TaskGyms for worker processes"""

    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self):
        return TaskGym(self.tasks)

def play_jobs(games, jobs, choose):
    """Plays every job once on the game slots"""
    queue = JobQueue(jobs)
    n = games.n
    playing, current = [False] * n, [None] * n
    obs, masks = [None] * n, None
    out = {}
    while len(queue) or any(playing):
        live = [i for i in range(n) if playing[i]]
        actions = [None] * n
        if live:
            picked = choose([obs[i] for i in live], masks[live])
            for k, i in enumerate(live):
                actions[i] = int(picked[k])
        offer = queue.offer(playing)
        obs, masks, rewards, dones, infos = games.step(actions, offer)
        for i in live:
            if dones[i]:
                out[current[i]] = infos[i]
        for i, job in queue.taken(offer, playing, dones).items():
            current[i] = job
        playing = [o is not None for o in obs]
    return out

def as_results(rows, infos, first_job):
    """One result dict per eval row"""
    res = []
    for j, r in enumerate(rows):
        info = infos[first_job + j]
        res.append({"score": info["score"], "kind": kind(info), "A": r["A"]["score"], "B": r["B"]["score"],
                    "plan": info["plan"]})
    return res

def keep_best(tries):
    """Finds, per task, the result with the highest score"""
    best = list(tries[0])
    for results in tries[1:]:
        for i, x in enumerate(results):
            if x["score"] > best[i]["score"]:
                best[i] = x
    return best

def compare(results):
    """Summary of one result per task vs Baselines A and B"""
    solvable = [x for x in results if x["A"] > 0]
    return {"score": mean([x["score"] for x in solvable]),
            "beats_A": mean([x["score"] > x["A"] for x in solvable]),
            "ties_A": mean([x["score"] == x["A"] for x in solvable]),
            "beats_B": mean([x["score"] >= x["B"] for x in solvable]),
            "invalid": mean([x["kind"] != "valid" for x in solvable]),
            "solvable": len(solvable),
            "gate_A": mean([x["score"] >= x["A"] - TIE for x in solvable])}

def save_plans(path, rows, greedy_res, best_res):
    """Write the RL's layouts for every eval task to a JSON file"""
    out = []
    for r, x, b in zip(rows, greedy_res, best_res):
        out.append({"seed": r["seed"],
                    "greedy": {"plan": x["plan"], "score": x["score"]},
                    "best": {"plan": b["plan"], "score": b["score"]}})
    Path(path).write_text(json.dumps(out, indent=1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--slots", type=int, default=48)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    load_checkpoint(args.checkpoint, policy, device=device)
    policy.eval()
    rows = load_eval(args.stage)
    n = len(rows)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def greedy(obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, greedy=True)[0]

    def sampled(obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, gen=gen)[0]

    t0 = time.time()
    with ProcGames(EvalTasks([r["task"] for r in rows]), args.slots, args.workers) as games:
        g = as_results(rows, play_jobs(games, list(range(n)), greedy), 0)
        infos = play_jobs(games, list(range(n, n * (args.k + 1))), sampled)
    tries = [as_results(rows, infos, n * (t + 1)) for t in range(args.k)]
    best = keep_best(tries)
    print(f"{n} tasks, {args.k} samples each, {time.time() - t0:.0f} s\n")

    print("task                       A      B      greedy  best-of-k")
    for r, x, b in zip(rows, g, best):
        name = r["target"].replace("Desc_", "").replace("_C", "")
        t = r["task"]
        print(f"{name:9s} {t['rate']:>5}/min {t['width']:>2}x{t['depth']:<3} {x['A']:6.3f} {x['B']:6.3f}  "
              f"{x['score']:6.3f}  {b['score']:6.3f}")

    if args.save:
        save_plans(args.save, rows, g, best)
        print(f"saved {args.save}")

    print("\non the tasks Baseline A can solve:")
    for name, res in [("greedy", g), (f"best-of-{args.k}", best)]:
        s = compare(res)
        print(f"{name:12s} mean {s['score']:.3f}  beats A {s['beats_A']:.0%}  ties A {s['ties_A']:.0%}  >= A-{TIE} {s['gate_A']:.0%}  "
              f">= B {s['beats_B']:.0%}  not valid {s['invalid']:.0%}  ({s['solvable']} tasks)")

if __name__ == "__main__":
    main()