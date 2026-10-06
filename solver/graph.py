import math
from solver.gamedata import items, recipe_by_id
from solver.lp_solve import solve_lp


def round_machines(count, mode="even"):
    n = math.ceil(count - 1e-9)
    if mode == "even":
        return [count / n] * n
    if mode == "last":
        return [1.0] * (n - 1) + [count - (n - 1)]
    raise ValueError(f"Unknown mode: {mode}")


BELT_CAPACITY = [60, 120, 270, 480, 780, 1200] # Mk1 to Mk6, rates in items/min

def pick_belt(rate, max_tier=6):
    for tier in range(1, max_tier + 1):
        if rate <= BELT_CAPACITY[tier - 1] + 1e-9:
            return tier, 1
    cap = BELT_CAPACITY[max_tier - 1]
    return max_tier, math.ceil(rate / cap - 1e-9)

def build_graph(target, rate, alternates=None, clock_mode="even", max_tier=6):
    machines, raw, edges = solve_lp(target, rate, alternates)

    nodes = {}
    for rid, count in machines.items():
        nodes[rid] = {
            "machine": recipe_by_id[rid]["machine"],
            "clocks": round_machines(count, clock_mode),
        }

    graph_edges = []
    for (src, dst, item), r in edges.items():
        solid = items[item]["form"] == "RF_SOLID"
        tier, belts = pick_belt(r, max_tier) if solid else (None, None)
        graph_edges.append({
            "from": src, "to": dst, "item": item, "rate": r,
            "belt_tier": tier, "belts": belts,
        })

    return {"target": target, "rate": rate, "nodes": nodes, "edges": graph_edges}