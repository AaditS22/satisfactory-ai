import numpy as np

from layout.connect import chains, distance
from layout.route import EMPTY, BLOCKED, BELT_X, BELT_Y
from layout.fastroute import route_fast as route  
from layout.world import EMPTY as NO_OWNER

ROUNDS = 8
HIST = 4

def base_kind(env):
    """Generates the routing grid for a finished episode"""
    w = env.world
    blocked = (w.owner != NO_OWNER) | w.reserved
    return np.where(blocked, BLOCKED, EMPTY).astype(np.int8)

def links(env):
    """Finds every belt link the episode needs"""
    ch = chains(env)
    out = []

    for item in sorted(ch):
        cells = ch[item]["cells"]
        rates = ch[item]["rates"]

        for i in range(len(cells) - 1):
            out.append((item, cells[i], cells[i + 1], rates[i]))

    return out

def lay(kind, path):
    """Mark a routed belt into kind in place"""
    for i in range(1, len(path) - 1):
        f, x, y = path[i]
        pf, px, py = path[i - 1]
        nf, nx, ny = path[i + 1]

        if pf != f or nf != f:
            lo = min(pf, f, nf)
            hi = max(pf, f, nf)
            kind[lo:hi + 1, x, y] = BLOCKED
        elif kind[f, x, y] != EMPTY:
            kind[f, x, y] = BLOCKED
        elif py == ny:
            kind[f, x, y] = BELT_X
        elif px == nx: 
            kind[f, x, y] = BELT_Y
        else: 
            kind[f, x, y] = BLOCKED

def route_round(base, order, extra):
    """Route every link in order, laying each one that succeeds"""
    kind = base.copy()
    paths = {}
    failed = []
    cost = 0

    for L in order:
        r = route(kind, L[1], L[2], extra)
        if r is None:
            failed.append(L)
        else:
            path, c = r
            lay(kind, path)
            paths[L] = path
            cost += c

    return paths, failed, kind, cost

def blame(base, kind, failed, hist, h):
    """Add toll h to hist on every cell that a failed link wanted but another belt used"""
    for L in failed:
        r = route(base, L[1], L[2])

        if r is None:
            continue

        for c in r[0][1:-1]:
            if kind[c] != base[c]:
                hist[c] += h

def route_all(env, rounds=ROUNDS, h=HIST):
    """Route every link of a finished episode, retrying with tolls when some fail"""
    base = base_kind(env)
    hist = np.zeros(base.shape, dtype=np.int32)
    order = sorted(links(env), key=lambda L: distance(L[1], L[2]))

    for r in range(rounds):
        paths, failed, kind, cost = route_round(base, order, hist)
        if not failed:
            break

        blame(base, kind, failed, hist, h)
        order = failed + [L for L in order if L not in failed]

    return {"paths": paths, "failed": failed, "cost": cost, "rounds": r + 1, "kind": kind}