import argparse
import time

from layout.evaluate import load_eval, play_eval, TIE
from layout.runlog import mean

import torch
from layout.policy import Policy, obs_to_torch
from rl.checkpoint import load_checkpoint
from rl.collect import act

import json
from pathlib import Path
from layout.api import plan_of 

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
                    "greedy": {"plan": plan_of(x["gym"].env), "score": x["score"]},
                    "best": {"plan": plan_of(b["gym"].env), "score": b["score"]}})
    Path(path).write_text(json.dumps(out, indent=1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    load_checkpoint(args.checkpoint, policy, device=device)
    policy.eval()
    rows = load_eval(args.stage)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def greedy(policy, obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, greedy=True)[0]

    def sampled(policy, obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, gen=gen)[0]

    t0 = time.time()
    g = play_eval(policy, rows, greedy)
    tries = [play_eval(policy, rows, sampled) for _ in range(args.k)]
    best = keep_best(tries)
    print(f"{len(rows)} tasks, {args.k} samples each, {time.time() - t0:.0f} s\n")

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