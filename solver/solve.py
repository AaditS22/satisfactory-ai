import json
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "game_data.json"

with open(DATA_PATH) as f:
    data = json.load(f)
items, recipes = data["items"], data["recipes"]

default_recipe = {}
for r in recipes:
    if r["alternate"]:
        continue
    for out in r["outputs"]:
        if items.get(out, {}).get("name") == r["name"]:
            default_recipe[out] = r


def solve(target, rate):
    machines = defaultdict(float)
    raw = defaultdict(float)
    edges = defaultdict(float)

    def need(item, amount, consumer): 
        recipe = default_recipe.get(item)
        producer = recipe["id"] if recipe else "RAW" 
        edges[(producer, consumer, item)] += amount 
        if recipe is None:
            raw[item] += amount
            return
        count = amount / recipe["outputs"][item]
        machines[recipe["id"]] += count
        for inp, per_min in recipe["inputs"].items():
            need(inp, per_min * count, recipe["id"]) 

    need(target, rate, "OUTPUT")  
    return dict(machines), dict(raw), dict(edges) 


def show(target, rate):
    machines, raw, edges = solve(target, rate)
    recipe_by_id = {r["id"]: r for r in recipes}

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