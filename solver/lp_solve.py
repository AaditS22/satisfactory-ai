from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

from solver.gamedata import items, recipes, recipe_by_id

V1_MACHINES = {
    "Build_SmelterMk1_C",
    "Build_ConstructorMk1_C",
    "Build_AssemblerMk1_C",
    "Build_ManufacturerMk1_C",
    "Build_FoundryMk1_C",
}

RESOURCE_LIMITS = {
    "Desc_OreIron_C": 92100,
    "Desc_OreCopper_C": 36900,
    "Desc_Stone_C": 69300,
    "Desc_Coal_C": 42300,
    "Desc_OreGold_C": 15000,
    "Desc_RawQuartz_C": 13500,
    "Desc_Sulfur_C": 10800,
    "Desc_OreBauxite_C": 12300,
    "Desc_OreUranium_C": 2100,
    "Desc_SAM_C": 10200,
}

MACHINE_PENALTY = 1e-6
EPS = 1e-6

def is_solid(item):
    return items.get(item, {}).get("form") == "RF_SOLID"

def allowed_recipes(alternates):
    """alternates: None = every alternate allowed, otherwise a set of allowed alternate ids."""
    out = []
    for r in recipes:
        if r["machine"] not in V1_MACHINES:
            continue
        if not all(is_solid(i) for i in (*r["inputs"], *r["outputs"])):
            continue
        if r["alternate"] and alternates is not None and r["id"] not in alternates:
            continue
        out.append(r)
    return out

def solve_lp(target, rate, alternates=None):
    rs = allowed_recipes(alternates)
    raws = list(RESOURCE_LIMITS)

    all_items = sorted({i for r in rs for i in (*r["inputs"], *r["outputs"])} | set(raws) | {target})
    row = {item: n for n, item in enumerate(all_items)}

    n_r = len(rs)
    A = np.zeros((len(all_items), n_r + len(raws)))
    for c, r in enumerate(rs):
        for item, per_min in r["outputs"].items():
            A[row[item], c] += per_min
        for item, per_min in r["inputs"].items():
            A[row[item], c] -= per_min
    for k, item in enumerate(raws):
        A[row[item], n_r + k] = 1.0

    demand = np.zeros(len(all_items))
    demand[row[target]] = rate

    cost = np.concatenate([
        np.full(n_r, MACHINE_PENALTY),
        [1.0 / RESOURCE_LIMITS[i] for i in raws],
    ])

    res = linprog(cost, A_ub=-A, b_ub=-demand, bounds=(0, None), method="highs")
    if not res.success:
        raise ValueError(f"No solid-only factory can make {target}: {res.message}")

    x = res.x
    machines = {rs[c]["id"]: float(x[c]) for c in range(n_r) if x[c] > EPS}
    raw = {raws[k]: float(x[n_r + k]) for k in range(len(raws)) if x[n_r + k] > EPS}
    return machines, raw, split_edges(machines, raw, target, rate)    

def split_edges(machines, raw, target, rate):
    made = defaultdict(dict)  
    used = defaultdict(dict) 
    for rid, count in machines.items():
        for item, per_min in recipe_by_id[rid]["outputs"].items():
            made[item][rid] = per_min * count
        for item, per_min in recipe_by_id[rid]["inputs"].items():
            used[item][rid] = per_min * count
    for item, amount in raw.items():
        made[item]["RAW"] = amount
    used[target]["OUTPUT"] = rate

    edges = {}
    for item, producers in made.items():
        total = sum(producers.values())
        consumers = dict(used[item])
        surplus = total - sum(consumers.values())
        if surplus > EPS:
            consumers["SURPLUS"] = surplus
        for src, p in producers.items():
            for dst, need in consumers.items():
                edges[(src, dst, item)] = need * p / total
    return edges

def show(target, rate, alternates=None):
    machines, raw, edges = solve_lp(target, rate, alternates)

    def label(x):
        return recipe_by_id[x]["name"] if x in recipe_by_id else x

    for rid, count in machines.items():
        r = recipe_by_id[rid]
        print(f"{count:6.2f} × {r['machine']:<28} {r['name']}")
    print("Raw:")
    for item, amount in raw.items():
        print(f"{amount:8.1f}/min  {items[item]['name']}")
    print("Edges:")
    for (src, dst, item), r in edges.items():
        print(f"{r:8.1f}/min  {items[item]['name']:<24} {label(src)} -> {label(dst)}")