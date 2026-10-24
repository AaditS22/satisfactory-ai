import json
from pathlib import Path

import numpy as np

from layout.gym_env import LayoutGym
from layout.runlog import kind, mean
from layout.view import draw
from layout.wire import route_all
from solver.gamedata import items

EVAL_PATH = Path("data/eval_set.json")
TIE = 0.005

def load_eval(stage, path=EVAL_PATH):
    """Loads the fixed test tasks of one curriculum stage"""
    rows = [r for r in json.loads(Path(path).read_text()) if r["stage"] == stage]
    for r in rows:
        if r["task"].get("alternates") is not None:
            r["task"]["alternates"] = frozenset(r["task"]["alternates"])
    return rows

def play_eval(policy, rows, choose):
    """Play every task once, all side by side"""
    gyms = [LayoutGym() for _ in rows]
    obs = [g.reset(options={"task": r["task"]})[0] for g, r in zip(gyms, rows)]
    infos = [None] * len(rows)
    playing = list(range(len(rows)))
    while playing:
        masks = np.stack([gyms[i].action_masks() for i in playing])
        actions = choose(policy, [obs[i] for i in playing], masks)
        still = []
        for i, a in zip(playing, actions):
            obs[i], reward, term, trunc, info = gyms[i].step(a)
            if term or trunc:
                infos[i] = dict(info, score=reward)
            else:
                still.append(i)
        playing = still
    return [{"score": info["score"], "kind": kind(info), "A": r["A"]["score"], "B": r["B"]["score"], "gym": g}
            for info, r, g in zip(infos, rows, gyms)]

def eval_stats(results):
    """One row of eval numbers summarizing model performance vs baselines"""
    solvable = [x for x in results if x["A"] > 0]
    return {"score": mean([x["score"] for x in results]),
            "A": mean([x["A"] for x in results]),
            "B": mean([x["B"] for x in results]),
            "solv_score": mean([x["score"] for x in solvable]),
            "solv_A": mean([x["A"] for x in solvable]),
            "gate_A": mean([x["score"] >= x["A"] - TIE for x in solvable]),
            "valid_score": mean([x["score"] for x in results if x["kind"] == "valid"]),
            "failed": mean([x["kind"] == "failed" for x in results]),
            "invalid": mean([x["kind"] == "invalid" for x in results])}

def best_order(results, k):
    """Indices of the k highest-scoring results"""
    return sorted(range(len(results)), key=lambda i: -results[i]["score"])[:k]

def save_best(results, folder, k=10):
    """Draw the k best layouts"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for rank, i in enumerate(best_order(results, k)):
        x = results[i]
        env = x["gym"].env
        title = (f"{items[env.graph['target']]['name']} at {env.graph['rate']}/min, "
                 f"{len(env.world.blocks)} blocks ({x['kind']})\n"
                 f"RL {x['score']:.3f}   Baseline A {x['A']:.3f}   Baseline B {x['B']:.3f}")
        routes = route_all(env) if x["kind"] != "failed" else None
        path = folder / f"best_{rank}.png"
        draw(env, path, title=title, routes=routes)
        paths.append(path)
    return paths

def evaluate(policy, rows, folder, choose, k=10):
    """Play the eval tasks, summarise them and draw the k best"""
    results = play_eval(policy, rows, choose)
    pngs = save_best(results, folder, k) if k > 0 else []
    return eval_stats(results), pngs