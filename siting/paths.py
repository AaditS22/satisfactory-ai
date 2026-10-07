import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from siting.ground import analyse
from siting.terrain import CELLS, CHUNK, STEP, TerrainCache
from siting.transport import _supply_cost, node_rate, pick_nodes

KIND_COST = np.array([np.inf, 2.0, 4.0, 1.0, 1.0, 1.0])
UNKNOWN_COST = 1.0 
REACH = 1.5     
MARGIN = 200.0   

NEIGHBOURS = ((1, 0), (0, 1), (1, 1), (1, -1)) 

def cost_grid(g, cache, box):
    cost = KIND_COST[g.kind]
    t = g.terrain
    unknown = False
    for i, j in cache.chunks_for(*box):
        if not cache.has(i, j):
            a = round((i * CHUNK + STEP / 2 - t.x0) / STEP)
            b = round((j * CHUNK + STEP / 2 - t.y0) / STEP)
            cost[a:a + CELLS, b:b + CELLS] = UNKNOWN_COST
            unknown = True
    return cost, unknown

def grid_graph(cost):
    nx, ny = cost.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, weights = [], [], []
    for dx, dy in NEIGHBOURS:
        a = (slice(max(0, -dx), nx - max(0, dx)), slice(max(0, -dy), ny - max(0, dy))) 
        b = (slice(max(0, dx), nx - max(0, -dx)), slice(max(0, dy), ny - max(0, -dy)))   
        w = STEP * math.hypot(dx, dy) * (cost[a] + cost[b]) / 2
        ok = np.isfinite(w)  
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
        weights.append(w[ok])
    n = nx * ny
    return coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

def path_haul(site, nodes, demand, cache=None):
    cache = cache or TerrainCache()
    cx, cy = site["centre"]

    straight = pick_nodes(site["centre"], nodes, demand)
    reach = max(d for _, _, d in straight) * REACH + MARGIN
    box = (cx - reach, cy - reach, cx + reach, cy + reach)
    g = analyse(cache.region(*box))
    t = g.terrain
    cost, unknown = cost_grid(g, cache, box)
    nx, ny = cost.shape

    def cell(x, y):
        return round((x - t.x0) / STEP), round((y - t.y0) / STEP)

    start = np.ravel_multi_index(cell(cx, cy), cost.shape)
    dist, pred = dijkstra(grid_graph(cost), directed=False, indices=start, return_predecessors=True)

    total, used = 0.0, []
    for resource, amount in demand.items():
        mine, d = [], []
        for n in nodes:
            if n["resource"] != resource or n["occupied"]:
                continue
            ix, iy = cell(*n["pos"][:2])
            if not (0 <= ix < nx and 0 <= iy < ny):
                continue
            straight_m = math.hypot(n["pos"][0] - cx, n["pos"][1] - cy)
            path_m = dist[ix * ny + iy]
            if np.isfinite(path_m):
                mine.append((n, ix * ny + iy))
                d.append(max(path_m, straight_m))
        rates = np.array([node_rate(n) for n, _ in mine])
        if rates.sum() < amount:
            return math.inf, [], unknown
        cost_r, order, take = _supply_cost(np.array(d), rates, amount)
        total += cost_r
        for k, amt in zip(order, take):
            if amt > 0:
                used.append((mine[k][0], float(amt), float(d[k]), _route(pred, start, mine[k][1], t, ny)))
    return total / sum(demand.values()), used, unknown

def _route(pred, start, end, t, ny):
    points, c = [], end
    while c != start and c >= 0:
        points.append(t.cell_centre(c // ny, c % ny))
        c = pred[c]
    points.append(t.cell_centre(start // ny, start % ny))
    return points

def main():
    from solver.lp_solve import solve_lp
    from siting.transport import load_nodes
    target, rate = sys.argv[1], float(sys.argv[2])
    x, y = float(sys.argv[3]), float(sys.argv[4])
    _, demand, _ = solve_lp(target, rate, alternates=set())
    haul, used, unknown = path_haul({"centre": (x, y)}, load_nodes(), demand)
    print(f"path haul {haul:.0f} m" + ("  (some ground not sampled yet)" if unknown else ""))
    for n, amt, d, _ in used:
        print(f"  {n['resource']:<20} {amt:6.1f}/min  {d:6.0f} m by path")

if __name__ == "__main__":
    main()