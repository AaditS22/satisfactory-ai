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

    def need(item, amount):
        recipe = default_recipe.get(item)
        if recipe is None:
            raw[item] += amount
            return
        count = amount / recipe["outputs"][item]
        machines[recipe["id"]] += count
        for inp, per_min in recipe["inputs"].items():
            need(inp, per_min * count)

    need(target, rate)
    return dict(machines), dict(raw)


def show(target, rate):
    machines, raw = solve(target, rate)
    recipe_by_id = {r["id"]: r for r in recipes}
    for rid, count in machines.items():
        r = recipe_by_id[rid]
        print(f"{count:6.2f} × {r['machine']:<28} {r['name']}")
    print("Raw:")
    for item, amount in raw.items():
        print(f"{amount:8.1f}/min  {items[item]['name']}")