import numpy as np
from layout.connect import ends_for
from layout.world import end_cells
from layout.tasks import GRID_MAX
 
S = 2
COARSE = GRID_MAX // S
COST_SCALE = 64.0
COST_CAP = 4.0

def port_cost(ex, ey, sources, X, Y):
    """Creates a distance map for one port"""
    if not sources:
        return np.zeros((X, Y), dtype=np.float32)

    xs = np.arange(X)[:, None]
    ys = np.arange(Y)[None, :]
    best = np.full((X, Y), np.inf)
    for sx, sy in sources:
        d = np.abs(xs + ex - sx) + np.abs(ys + ey - sy)
        best = np.minimum(best, d)
    return best.astype(np.float32)

def square_order(s=S):
    """Cells inside one s x s square in the order gym_env.fine_cell tries them"""
    cells = [(dx, dy) for dx in range(s) for dy in range(s)]
    return sorted(cells, key=lambda c: (c[0] + c[1], c[0]))
 
def block_metres(env, block):
    """Estimates the belt metres for ports"""
    _, X, Y = env.world.owner.shape
    ends = end_cells(block)
    total = np.zeros((X, Y), dtype=np.float32)
    for i, inp in enumerate(block["inputs"]):
        prods = ends_for(env, inp["item"])[0]
        total += port_cost(*ends[i], [(x, y) for _, x, y in prods], X, Y)
    cons = ends_for(env, block["output"]["item"])[2]
    total += port_cost(*ends[-1], [(x, y) for _, x, y in cons], X, Y)
    return total
 
def snap(values, legal):
    """Shrinks the map and keeps the value the action would land on (or 0 if illegal)"""
    X, Y = legal.shape
    v = np.zeros((GRID_MAX, GRID_MAX), dtype=np.float32)
    m = np.zeros((GRID_MAX, GRID_MAX), dtype=bool)
    v[:X, :Y] = values
    m[:X, :Y] = legal
    out = np.zeros((COARSE, COARSE), dtype=np.float32)
    filled = np.zeros((COARSE, COARSE), dtype=bool)
    for dx, dy in square_order():
        here = m[dx::S, dy::S] & ~filled
        out[here] = v[dx::S, dy::S][here]
        filled |= here
    return out
 
def block_cost(env, n, fine):
    """The cost observation for placing a block of n machines"""
    R, F = fine.shape[0], fine.shape[1]
    out = np.zeros((R, F, COARSE, COARSE), dtype=np.float32)
    for rot in range(R):
        metres = block_metres(env, env.block(n, rot))
        for f in range(F):
            out[rot, f] = snap(metres, fine[rot, f])
    return np.minimum(out / COST_SCALE, COST_CAP)