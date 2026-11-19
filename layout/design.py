import argparse
import time

import numpy as np
import torch

from layout.anneal import baseline_b
from layout.api import evaluate
from layout.baseline import baseline_a
from layout.best_of import EvalTasks, play_jobs
from layout.env import LayoutEnv, raw_items
from layout.obs import MAX_NODES, MAX_EDGES
from layout.policy import Policy, obs_to_torch
from layout.runlog import kind
from layout.tasks import FLOORS, GRID_MAX, MAX_ASPECT, MAX_MACHINES, machine_count, min_sides, one_belt, scale, site_size
from layout.view import draw_compare
from rl.checkpoint import load_checkpoint
from rl.collect import act
from rl.games import ProcGames
from siting.choose import min_floor_area
from solver.gamedata import items
from solver.graph import build_graph
from solver.lp_solve import solve_lp

MODELS = [f"models/stage{s}.pt" for s in range(1, 6)]  # the specialist of every stage, in stage order
MODES = ["all", "greedy", "scout"]
K = 64      # samples per task
SCOUT = 8   # samples per specialist in the scout look
SEED = 0
WORKERS = 16
SLOTS = 48

def find_item(name):
    """Item id from an in-game name ('Smart Plating') or an id ('Stator', 'Desc_Stator_C')"""
    for key in (name, f"Desc_{name}_C"):
        if key in items:
            return key
    for key, item in items.items():
        if item["name"].lower() == name.lower():
            return key
    raise ValueError(f"unknown item '{name}'")

def check_fits(task):
    """Raises ValueError if the task is outside what the models can play"""
    if max(task["width"], task["depth"]) > GRID_MAX:
        raise ValueError(f"the site can be at most {GRID_MAX} x {GRID_MAX} m")
    graph = build_graph(task["target"], task["rate"], task["alternates"])
    if len(graph["nodes"]) + len(raw_items(graph)) + 1 > MAX_NODES or len(graph["edges"]) > MAX_EDGES:
        raise ValueError("the production graph is too big for the network")
    if not one_belt(task):
        raise ValueError("an item needs more than one belt at this rate, try a lower rate")
    env = LayoutEnv()
    env.reset(task)
    if env.done:
        raise ValueError("the first machines don't fit on this site, try a bigger one")

def make_task(target, rate, site=None):
    """Task for target at rate (any alternate recipe allowed) on a 'WxD' m site, picked like the training tasks when left out"""
    alternates = None
    unit, _, _ = solve_lp(target, 1, alternates)
    machines = scale(unit, rate)
    if machine_count(machines) > MAX_MACHINES or min_floor_area(machines) > GRID_MAX * GRID_MAX:
        raise ValueError(f"too many machines at {rate}/min, try a lower rate")

    rng = np.random.default_rng(SEED)
    side = int(rng.integers(4))
    if site is None:
        area = min_floor_area(machines) * rng.uniform(1.0, 4.0)
        width, depth = site_size(area, rng.uniform(1.0, MAX_ASPECT))
        long_side, short_side = min_sides(unit)
        width, depth = max(width, long_side), max(depth, short_side)
    else:
        width, depth = (int(v) for v in site.lower().split("x"))

    task = {"target": target, "rate": rate, "width": int(width), "depth": int(depth),
            "floors": FLOORS, "side": side, "alternates": alternates}
    check_fits(task)
    return task

def task_name(task):
    """Short description of a task"""
    return f"{items[task['target']]['name']} at {task['rate']}/min on a {task['width']} x {task['depth']} m site"

def load_models(device):
    """The specialist of every stage, in stage order"""
    models = []
    for path in MODELS:
        policy = Policy().to(device)
        load_checkpoint(path, policy, device=device)
        policy.eval()
        models.append(policy)
    return models

def play(games, policy, n, gen=None, task=0, n_tasks=1):
    """Plays task number "task" (of the n_tasks the games hold) n times with one model, greedily when gen is None"""
    def choose(obs_list, masks):
        return act(policy, obs_list, masks, obs_to_torch, gen=gen, greedy=gen is None)[0]
    return list(play_jobs(games, [task + j * n_tasks for j in range(n)], choose).values())

def best(tries):
    """The game with the highest score"""
    return max(tries, key=lambda info: info["score"])

def design(games, models, mode, gen, k=K, scout=SCOUT, task=0, n_tasks=1):
    """Plays the task with the specialists in one mode; gives (best game, its stage, first look per stage, games played)"""
    first_n = {"all": k, "greedy": 1, "scout": scout}[mode]
    more = {"all": 0, "greedy": k, "scout": k - scout}[mode]

    first = [play(games, m, first_n, None if mode == "greedy" else gen, task, n_tasks) for m in models]
    looks = [best(f) for f in first]
    pick = max(range(len(models)), key=lambda s: looks[s]["score"])
    tries = first[pick] + play(games, models[pick], more, gen, task, n_tasks)
    return best(tries), pick + 1, looks, first_n * len(models) + more

def main():
    p = argparse.ArgumentParser(description="design a layout for any supported item and rate with the stage specialists")
    p.add_argument("item", help="in-game name or id, e.g. Stator or \"Smart Plating\"")
    p.add_argument("rate", type=float, help="items per minute")
    p.add_argument("--site", default=None, help="site size in m, e.g. 64x48 (default: picked like the training tasks)")
    p.add_argument("--mode", choices=MODES, default="scout")
    p.add_argument("--b", action="store_true", help="also run Baseline B (slow) for comparison")
    p.add_argument("--out", default=None, help="save a picture of the baselines and the agent's layout here")
    args = p.parse_args()

    rate = int(args.rate) if args.rate.is_integer() else args.rate
    try:
        task = make_task(find_item(args.item), rate, args.site)
    except ValueError as e:
        p.error(str(e))
    print(f"{task_name(task)}, alternate recipes allowed\n")

    t0 = time.time()
    a = baseline_a(task)
    time_a = time.time() - t0
    if args.b:
        t0 = time.time()
        b = baseline_b(task)
        time_b = time.time() - t0

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    gen = torch.Generator(device=device).manual_seed(SEED)
    with ProcGames(EvalTasks([task]), SLOTS, WORKERS) as games:
        time_start = time.time() - t0
        t0 = time.time()
        agent, stage, looks, n = design(games, models, args.mode, gen)
        time_agent = time.time() - t0

    look = "greedy" if args.mode == "greedy" else f"best of {K if args.mode == 'all' else SCOUT}"
    print(f"first look ({look}):  " + "  ".join(f"stage {s + 1} {x['score']:.3f}" for s, x in enumerate(looks)))
    print(f"picked the stage {stage} specialist, {n} games in all\n")

    print(f"Baseline A (greedy heuristic)     {a['score']:6.3f}  {time_a:6.2f} s")
    if args.b:
        print(f"Baseline B (simulated annealing)  {b['score']:6.3f}  {time_b:6.2f} s")
    print(f"{'RL agent (' + args.mode + ')':34s}{agent['score']:6.3f}  {time_agent:6.2f} s  {kind(agent)}")
    print(f"\n(+ {time_start:.1f} s to load the models and start the workers, paid once per run)")

    if args.out:
        rows = [("Greedy heuristic (Baseline A)", evaluate(task, a["plan"]), False)]
        if args.b:
            rows.append(("Simulated annealing (Baseline B)", evaluate(task, b["plan"]), False))
        rows.append((f"RL agent ({args.mode})", evaluate(task, agent["plan"]), True))
        draw_compare(rows, args.out, task_name(task))
        print(f"saved {args.out}")

if __name__ == "__main__":
    main()