import math
from layout.env import MAX_TIER
from solver.graph import BELT_CAPACITY

FLOOR_COST = 16

def distance(a, b):
    """Gets the belt distance between two cells"""
    fa, xa, ya = a
    fb, xb, yb = b

    return abs(xa - xb) + abs(ya - yb) + FLOOR_COST * abs(fa - fb)

def nearest(cell, cells, used):
    """Finds the index of the nearest unused cell to another"""
    best = None
    best_d = math.inf

    for i, c in enumerate(cells):
        if i in used:
            continue

        d = distance(c, cell)
        if d < best_d:
            best = i
            best_d = d

    return best        

def chain_order(prods, cons):
    """Orders an item's chain, producers and consumers"""
    start = None
    best = -1

    for i, p in enumerate(prods):
        near_cons = nearest(p, cons, set())
        dist = distance(p, cons[near_cons])

        if dist > best:
            best = dist
            start = i

    p_order = [start]
    used = {start}

    while len(p_order) < len(prods):
        i = nearest(prods[p_order[-1]], prods, used)
        p_order.append(i)
        used.add(i)

    c_order = []
    used = set()
    cur = prods[p_order[-1]]

    while len(c_order) < len(cons):
        j = nearest(cur, cons, used)
        c_order.append(j)
        used.add(j)
        cur = cons[j]

    return(p_order, c_order)

def ends_for(env, item):
    """All producer and consumer cells of one item in a finished episode"""
    prods = []
    p_rates = []
    cons = []
    c_rates = []

    for b in env.world.blocks:
        out =  b["block"]["output"]

        if out["item"] == item:
            cell = (b["floor"], *b["ends"][-1])
            prods.append(cell)
            p_rates.append(out["rate"])

        for i, j in enumerate(b["block"]["inputs"]):
            if j["item"] == item:
                cons.append((b["floor"], *b["ends"][i]))
                c_rates.append(j["rate"])

    if item == env.graph["target"]:
        cons.append((0, *env.exit))
        c_rates.append(env.graph["rate"])

    if item in env.entries:
        prods.append((0, *env.entries[item]))
        p_rates.append(sum(c_rates))

    return prods, p_rates, cons, c_rates

def trunk_rates(prates, crates, p_order, c_order):
    """Get rates on each of the trunk links in chain order"""
    flow = 0
    rates = []

    for i in p_order:
        flow += prates[i]
        rates.append(flow)

    for j in c_order:
        flow -= crates[j]
        rates.append(flow)

    return rates[:-1]

def chains(env):
    """{item: {"cells": chain-ordered cells, "rates": trunk_rates, "n_prod": number of producers}}.

    One chain per item in the graph's edges. Raises ValueError unless the episode is done and not failed.
    """
    if not env.done or env.failed:
        raise ValueError("Episode failed or incomplete")

    items = {e["item"] for e in env.graph["edges"]}
    res = {}

    for item in items:
        prods, prates, cons, crates = ends_for(env, item)
        p_order, c_order = chain_order(prods, cons)
        cells = [prods[i] for i in p_order] + [cons[j] for j in c_order]

        rates = trunk_rates(prates, crates, p_order, c_order)
        assert max(rates) <= BELT_CAPACITY[MAX_TIER - 1] + 1e-9
        res[item] = {"cells": cells, "rates": rates, "n_prod": len(prods)}

    return res

def added_belt(env):
    """Calculates the estimated meters of belt the newest block adds"""
    b = env.world.blocks[-1]
    total = 0
    for i, inp in enumerate(b["block"]["inputs"]):
        end = (b["floor"], *b["ends"][i])
        prods = [(o["floor"], *o["ends"][-1]) for o in env.world.blocks[:-1]
                 if o["block"]["output"]["item"] == inp["item"]]
        if inp["item"] in env.entries:
            prods.append((0, *env.entries[inp["item"]]))
        if prods:
            total += min(distance(end, p) for p in prods)
    if b["block"]["output"]["item"] == env.graph["target"]:
        total += distance((b["floor"], *b["ends"][-1]), (0, *env.exit))
    return total