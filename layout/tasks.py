import math

import numpy as np

from solver.lp_solve import allowed_recipes, solve_lp, RESOURCE_LIMITS
from siting.choose import min_floor_area
from layout.blocks import make_block

RATES = [1, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120] # items/min a player could ask for
MAX_MACHINES = 48
GRID_MAX = 96
FLOORS = 3
STEP = 8
MAX_ASPECT = 3.0

_targets = None  # cache for targets()

def round_up(v, step=STEP):
    """Smallest multiple of step that is >= v"""

    temp = v / step
    return int(math.ceil(temp - 1e-9) * step)

def site_size(area, aspect, cap=GRID_MAX):
    """Width and depth of a site with at least "area" m2 per floor"""

    width = min(round_up(math.sqrt(area * aspect)), cap)
    depth = min(round_up(area / width), cap)
    return (width, depth)

def all_alternates():
    """Sorted list of every alternate recipe id our machines can run"""

    recipes = allowed_recipes(None)
    alts = []
    for r in recipes:
        if r["alternate"] == True:
            alts.append(r["id"])

    return sorted(alts)        

def targets():
    """Sorted list of every item the LP can make when all alternates are allowed."""

    global _targets
    if _targets is None:
        outputs = set()
        for r in allowed_recipes(None):
            outputs.update(r["outputs"])

        found = []
        for out in sorted(outputs.difference(set(RESOURCE_LIMITS))):
            try:
                solve_lp(out, 1, None)
                found.append(out)
            except ValueError:
                pass

        _targets = found

    return _targets            

def random_alternates(rng):
    """A random set of alternate ids"""

    alts = all_alternates()
    p = rng.uniform()
    keep = rng.random(len(alts)) < p

    res = []
    for i, alt in enumerate(alts):
        if keep[i]:
            res.append(alt)

    return frozenset(res)        

def scale(machines, rate):
    """Scale each machine's numbers by rate"""

    scaled = {}
    for recipe in machines:
        scaled[recipe] = machines[recipe] * rate

    return scaled

def machine_count(machines):
    """Count total whole machines"""

    res = 0
    for num in machines.values():
        res += math.ceil(num - 1e-9)

    return res    

def valid_rates(unit_machines, cap=GRID_MAX):
    """Get a list of the valid rates for the item"""
    valid = []

    for r in RATES:
        m = scale(unit_machines, r)
        if machine_count(m) <= MAX_MACHINES and min_floor_area(m) <= cap * cap:
            valid.append(r)

    return valid

def min_sides(recipes):
    """Get the long and short site sides needed so a 1-machine block of every recipe fits in some rotation"""
    long_side = 0
    short_side = 0

    for rid in recipes:
        b = make_block(rid, 1, 1.0)
        a, c = sorted((b["w"] + 2, b["d"]))
        short_side = max(short_side, a)
        long_side = max(long_side, c)

    return (round_up(long_side), round_up(short_side))

def random_task(rng):
    """Generates a random task for training process"""
    options = targets()

    while True:
        target = options[rng.integers(len(options))]
        alternates = random_alternates(rng)

        try:
            unit, _, _ = solve_lp(target, 1, alternates)
            rates = valid_rates(unit)

            if len(rates) == 0:
                continue

            rate = rates[rng.integers(len(rates))]
            area = min_floor_area(scale(unit, rate)) * rng.uniform(1.0, 4.0)
            aspect = rng.uniform(1.0, MAX_ASPECT)
            width, depth = site_size(area, aspect)

            long_side, short_side = min_sides(unit)
            width = max(width, long_side)
            depth = max(depth, short_side)

            if rng.choice([True, False]):
                width, depth = depth, width

            return {"target": target, "rate": rate, "width": int(width), "depth": int(depth),
                "floors": FLOORS, "side": int(rng.integers(4)), "alternates": alternates}
        except ValueError:
            continue