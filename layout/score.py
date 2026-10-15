import numpy as np

from layout.world import EMPTY as NO_OWNER
from layout.wire import route_all
from layout.flow import flow

# Costs and weights
BELT_REF = 32 # metres of belt per link
LIFT_REF = 1.0 # lifts per link
CROSS_REF = 0.25 # crossings per link
SPRAWL_REF = 0.5 # (bbox / used - 1)
FLOOR_REF = 2 # (floors used - 1)
WEIGHTS = {"p_belt": 0.35, "p_lifts": 0.15, "p_crossings": 0.15, "p_sprawl": 0.2, "p_floors": 0.15}

def footprint(env):
    """Finds how much floor the blocks take up"""
    occupied = env.world.owner != NO_OWNER
    floors = 0
    used = 0
    bbox = 0

    for floor in occupied:
        if not floor.any():
            continue

        x_indices, y_indices = np.where(floor == True)
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()

        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1

        bbox += span_x * span_y
        used += floor.sum()
        floors += 1

    return {"bbox": int(bbox), "used": int(used), "floors": int(floors)}

def belt_stats(routes):
    """Finds the number of belts, lifts, and crossings"""
    links = routes["paths"]
    length = 0
    lifts = 0
    through = {}

    for link in links:
        path = links[link]

        for i in range(1, len(path)):
            prev = path[i - 1]
            cur = path[i]

            if prev[0] != cur[0]:
                lifts += 1

            if prev[1] != cur[1] or prev[2] != cur[2]:
                length += 1

        for i in range(1, len(path) - 1):
            p, c, n = path[i - 1], path[i], path[i + 1]
            if p[0] == c[0] == n[0] and (p[1] == n[1] or p[2] == n[2]):
                through.setdefault(c, set()).add(link)

    cross_count = 0
    for users in through.values():
        if len(users) > 1:
            cross_count += 1

    return {"length": length, "lifts": lifts, "crossings": cross_count}

def measure(env, routes, fl):
    """All raw score terms of a finished episode"""
    links = len(routes["paths"]) + len(routes["failed"])
    routed = len(routes["paths"]) / links
    fraction, passed = fl["fraction"], fl["passed"]

    res = {"links": links, "routed": routed, "fraction": fraction, "passed": passed, "edges": len(env.graph["edges"])}
    res.update(belt_stats(routes))
    res.update(footprint(env))

    return res

def parts(terms):
    """Calculates the 5 normalized pernalty parts"""
    p_belt = terms["length"] / terms["edges"] / BELT_REF
    p_lifts = terms["lifts"] / terms["edges"] / LIFT_REF
    p_crossings = terms["crossings"] / terms["edges"] / CROSS_REF
    p_sprawl = (terms["bbox"] / terms["used"] - 1) / SPRAWL_REF
    p_floors = (terms["floors"] - 1) / FLOOR_REF

    return {"p_belt": p_belt, "p_lifts": p_lifts, "p_crossings": p_crossings, 
            "p_sprawl": p_sprawl, "p_floors": p_floors}
 
def penalty(terms):
    """Calculates the sum of each part * the weight"""
    p = parts(terms)
    return sum(p[k] * WEIGHTS[k] for k in WEIGHTS)
 
def score(terms):
    """Gives a score based on whether the layout is valid or not"""
    if terms["passed"]:
        return 1.0 / (1.0 + penalty(terms))

    return -1 + 0.5 * terms["fraction"] + 0.5 * terms["routed"]
 
def placed_fraction(env):
    """Percent of machines placed so far"""
    placed = 0
    for b in env.world.blocks:
        placed += b["block"]["n"]

    needed = 0
    for d in env.graph["nodes"].values():
        needed += len(d["clocks"])

    return placed / needed
 
def score_episode(env, routes=None):
    """Gets the final score for a finished episode"""
    if not env.done:
        raise ValueError("Not finished yet")

    if env.failed:
        p = placed_fraction(env)
        return (-2 + p, {"placed": p})

    if routes is None:
        routes = route_all(env)
        
    terms = measure(env, routes, flow(env, routes)) 
    terms["placed"] = 1.0
    terms.update(parts(terms))
    terms["penalty"] = penalty(terms)

    return (score(terms), terms)