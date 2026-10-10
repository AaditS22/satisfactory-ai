import json
import math
import sys
from pathlib import Path

import numpy as np

from siting.ground import analyse
from siting.sites import MAX_SIDE, TOP_K, draw_sites, find_sites, room_bonus
from siting.terrain import CHUNK, STEP, TerrainCache, CELLS
from siting.transport import load_nodes, transport_cost, transport_map
from solver.gamedata import items, recipe_by_id
from solver.lp_solve import solve_lp
from siting.paths import path_haul

TRANSPORT_WEIGHT = 0.01
PACKING = 2.5
MAX_FLOORS = 3 
COARSE = 64.0 
MAX_SAMPLED = 60 
MAX_ANALYSED = 150  
MIN_SPACING = 300.0 
PATH_POOL = 15  
CATALOG = Path(__file__).parent.parent / "data" / "catalog.json"

def min_floor_area(machines):
    buildings = json.loads(CATALOG.read_text())["buildings"]
    area = 0.0
    for recipe_id, count in machines.items():
        b = buildings[recipe_by_id[recipe_id]["machine"]]
        size_x = b["clearance_max"][0] - b["clearance_min"][0]
        size_y = b["clearance_max"][1] - b["clearance_min"][1]
        area += math.ceil(count - 1e-9) * size_x * size_y
    return area * PACKING / MAX_FLOORS

def chunk_bounds(nodes, demand, max_bonus):
    xs_all = [n["pos"][0] for n in nodes]
    ys_all = [n["pos"][1] for n in nodes]
    i0, i1 = math.floor(min(xs_all) / CHUNK) - 1, math.floor(max(xs_all) / CHUNK) + 1
    j0, j1 = math.floor(min(ys_all) / CHUNK) - 1, math.floor(max(ys_all) / CHUNK) + 1
    per = int(CHUNK / COARSE)    
    xs = i0 * CHUNK + (np.arange((i1 - i0 + 1) * per) + 0.5) * COARSE
    ys = j0 * CHUNK + (np.arange((j1 - j0 + 1) * per) + 0.5) * COARSE

    haul = transport_cost(xs, ys, nodes, demand) / sum(demand.values()) 
    slack = COARSE * math.sqrt(2) / 2
    lowest = haul.reshape(i1 - i0 + 1, per, j1 - j0 + 1, per).min(axis=(1, 3)) - slack
    bounds = TRANSPORT_WEIGHT * lowest - max_bonus
    return {(i0 + a, j0 + b): float(bounds[a, b]) for a in range(bounds.shape[0]) for b in range(bounds.shape[1])}

def _rect(s):
    """A site's footprint in world metres: (x_min, x_max, y_min, y_max)."""
    cx, cy = s["centre"]
    return cx - s["w"] * STEP / 2, cx + s["w"] * STEP / 2, cy - s["d"] * STEP / 2, cy + s["d"] * STEP / 2

def _overlap_world(a, b):
    ax0, ax1, ay0, ay1 = _rect(a)
    bx0, bx1, by0, by1 = _rect(b)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1

def _too_close(a, b):
    (ax, ay), (bx, by) = a["centre"], b["centre"]
    return _overlap_world(a, b) or math.hypot(ax - bx, ay - by) < MIN_SPACING

def top_k(candidates, k=TOP_K, key="score"):
    result = []
    for s in sorted(candidates, key=lambda s: s[key]):
        if len(result) == k:
            break
        if not any(_too_close(s, p) for p in result):
            result.append(s)
    return result

def _path_score(s, nodes, demand, cache):
    if "path_score" not in s:
        s["path_haul"], s["routes"], s["unknown"] = path_haul(s, nodes, demand, cache)
        s["path_score"] = s["score"] - s["extra"] + TRANSPORT_WEIGHT * s["path_haul"]
    return s["path_score"]

def best_by_path(candidates, nodes, demand, cache, k=1):
    found = []
    for s in sorted(candidates, key=lambda s: s["score"]):
        if len(found) >= k and s["score"] >= found[k - 1]:
            break
        found = sorted(found + [_path_score(s, nodes, demand, cache)])
    return found[:k]

def search(target, rate, alternates=frozenset(), log=print):
    machines, demand, _ = solve_lp(target, rate, alternates=alternates)
    min_area = min_floor_area(machines)
    max_bonus = room_bonus(MAX_SIDE * MAX_SIDE * STEP * STEP, min_area)
    nodes = load_nodes()
    total = sum(demand.values())

    bounds = chunk_bounds(nodes, demand, max_bonus)
    order = sorted(bounds, key=bounds.get)
    log(f"{len(order)} chunks on the map; best possible score {bounds[order[0]]:.2f} (chunk {order[0]})")

    cache = TerrainCache()
    candidates, unchecked = _visit(order, bounds, cache, nodes, demand, total, min_area, log)
    best_by_path(candidates, nodes, demand, cache)  
    for s in top_k(candidates, PATH_POOL):  
        _path_score(s, nodes, demand, cache)
    checked = [s for s in candidates if "path_score" in s and math.isfinite(s["path_score"])]
    best = min((s["path_score"] for s in checked), default=math.inf)
    unproven = [c for c in unchecked if bounds[c] < best]
    return top_k(checked, key="path_score"), demand, machines, min_area, nodes, unproven

def _visit(order, bounds, cache, nodes, demand, total, min_area, log):
    bridge, home = None, None
    offline = False
    candidates = []
    unchecked = [] 
    analysed, sampled = 0, 0
    middle = (CELLS - 0.5, 2 * CELLS - 0.5, CELLS - 0.5, 2 * CELLS - 0.5)
    try:
        for n, chunk in enumerate(order):
            bar = min(best_by_path(candidates, nodes, demand, cache), default=math.inf)
            if bounds[chunk] >= bar:
                log(f"stopping: no unvisited chunk can beat the best site (next bound {bounds[chunk]:.2f} >= {bar:.2f})")
                break
            if analysed == MAX_ANALYSED:
                log(f"stopping at the MAX_ANALYSED cap ({MAX_ANALYSED})")
                unchecked += order[n:]
                break

            i, j = chunk
            box = ((i - 1) * CHUNK, (j - 1) * CHUNK, (i + 2) * CHUNK - 1, (j + 2) * CHUNK - 1)
            new = 0
            if any(not cache.has(*c) for c in cache.chunks_for(*box)):
                if offline or sampled >= MAX_SAMPLED:
                    unchecked.append(chunk)
                    continue
                try:
                    from executor.bridge import Bridge
                    if bridge is None:
                        bridge = Bridge(timeout=60.0)
                        home = bridge.player()["pos"]
                    centre = ((i + 0.5) * CHUNK, (j + 0.5) * CHUNK)
                    z_guess = min(nodes, key=lambda nd: math.hypot(nd["pos"][0] - centre[0], nd["pos"][1] - centre[1]))["pos"][2]
                    new = cache.ensure(bridge, *box, z_guess=z_guess, return_home=False,
                        log=lambda m: log(m) if "WARNING" in m else None)
                    sampled += new
                except Exception as e:
                    offline = True
                    unchecked.append(chunk)
                    log(f"  chunk {chunk}: can't sample terrain ({e}); using cached areas only from now on")
                    continue
                if sampled >= MAX_SAMPLED:
                    log(f"  sampling budget used up ({sampled} chunks); unsampled areas are skipped from now on")

            analysed += 1
            g = analyse(cache.region(*box))
            extra = TRANSPORT_WEIGHT * transport_map(g.terrain, nodes, demand) / total
            found = find_sites(g, min_area, extra_cost=extra, keep=middle)
            candidates += found
            note = f", sampled {new} new" if new else ""
            log(f"  chunk {chunk}: bound {bounds[chunk]:.2f}, best here {found[0]['score']:.2f}{note}" if found
                else f"  chunk {chunk}: bound {bounds[chunk]:.2f}, nothing fits{note}")
    finally:
        if home is not None:
            try:
                bridge.teleport((home[0], home[1], home[2] + 1), freeze=False)
            except Exception:
                log("could not bring the player home (is the game still running?)")
    return candidates, unchecked

def main():
    target, rate = sys.argv[1], float(sys.argv[2])
    sites, demand, machines, min_area, nodes, unproven = search(target, rate)

    print(f"\n{rate:g}/min of {items[target]['name']} needs:")
    for resource, amount in demand.items():
        print(f"  {amount:8.1f}/min  {items[resource]['name']}")
    print(f"machines: {sum(math.ceil(c - 1e-9) for c in machines.values())}, "
          f"minimum site area ~{min_area:.0f} m2 (up to {MAX_FLOORS} floors)")
    if not sites:
        print("no site found" + (f" ({len(unproven)} areas not sampled yet: open the game and run again)" if unproven else ""))
        return

    print(f"\nbest {len(sites)} sites on the map:")
    print(f"  {'#':>2} {'centre (x, y)':>18} {'size (m)':>9} {'base z':>7} {'fill':>6} {'water':>6} "
          f"{'line haul':>9} {'path haul':>9} {'score':>6}")
    for n, s in enumerate(sites, 1):
        size = f"{s['w'] * STEP:.0f}x{s['d'] * STEP:.0f}"
        flag = " *" if s["unknown"] else ""
        print(f"  {n:>2} {s['centre'][0]:>8.0f}, {s['centre'][1]:>7.0f} {size:>9} {s['base']:>7.1f} "
              f"{s['fill']:>6.1f} {100 * s['water']:>5.0f}% {s['extra'] / TRANSPORT_WEIGHT:>7.0f} m "
              f"{s['path_haul']:>7.0f} m {s['path_score']:>6.2f}{flag}")
    if any(s["unknown"] for s in sites):
        print("  * some ground around this site isn't sampled yet; its path haul may be too low")

    if unproven:
        print(f"\nNOT proven best: {len(unproven)} unsampled chunk(s) could still beat #1. "
              f"Run again with the game open to sample more.")
    else:
        print("\nproven: no other place on the map can beat #1 under this scoring (#2-#5 are not proven)")

    best = sites[0]
    print(f"\nnodes feeding site #1:")
    for node, amount, dist, _ in best["routes"]:
        line = math.hypot(node["pos"][0] - best["centre"][0], node["pos"][1] - best["centre"][1])
        print(f"  {items[node['resource']]['name']:<12} {node['purity']:<7} {amount:6.1f}/min  "
              f"{dist:6.0f} m by path ({line:.0f} m straight)")

    cx, cy = best["centre"]
    view = 800.0
    cache = TerrainCache()
    g = analyse(cache.region(cx - view, cy - view, cx + view, cy + view))
    t = g.terrain
    shown, hidden = [], []
    for rank, s in enumerate(sites, 1):
        ix = round((s["centre"][0] - s["w"] * STEP / 2 + STEP / 2 - t.x0) / STEP)
        iy = round((s["centre"][1] - s["d"] * STEP / 2 + STEP / 2 - t.y0) / STEP)
        if 0 <= ix and 0 <= iy and ix + s["w"] <= t.z.shape[0] and iy + s["d"] <= t.z.shape[1]:
            shown.append({**s, "ix": ix, "iy": iy, "rank": rank})
        else:
            hidden.append(f"#{rank}")
    links = [route for _, _, _, route in best["routes"]]
    out = draw_sites(g, shown, f"{rate:g}/min {items[target]['name']}: best site on the map, nodes for #1",
                     Path(".maps") / f"choose_{target}.png", links=links)
    print(f"saved {out}")
    if hidden:
        print(f"not on the map (too far from #1): {', '.join(hidden)}")

if __name__ == "__main__":
    main()