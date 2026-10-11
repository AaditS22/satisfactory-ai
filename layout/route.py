import heapq
import itertools

from layout.connect import FLOOR_COST, distance

EMPTY, BLOCKED, BELT_X, BELT_Y = 0, 1, 2, 3
STEP, TURN, CROSS, LIFT = 1, 2, 6, 8
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
NO_DIR = 4

def neighbours(kind, state, goal):
    """Every legal next state from state with the cost of the move"""
    F, X, Y = kind.shape
    f, x, y, d = state
    here = kind[f, x, y]
    out = []

    for nd, (dx, dy) in enumerate(DIRS):
        if d != NO_DIR and nd == (d + 2) % 4:
            continue

        if here in (BELT_X, BELT_Y) and nd != d:
            continue

        nx = x + dx
        ny = y + dy

        if not (0 <= nx < X and 0 <= ny < Y):
            continue

        cost = STEP
        if d != NO_DIR and nd != d:
            cost += TURN

        k = kind[f, nx, ny]
        if (f, nx, ny) == goal:
            pass
        elif k == EMPTY:
            pass
        elif (k == BELT_X and nd in (1, 3)) or (k == BELT_Y and nd in (0, 2)):
            cost += CROSS
        else:
            continue

        out.append(((f, nx, ny, nd), cost))

    if d != NO_DIR and here == EMPTY:
        for nf in range(F):
            if nf == f:
                continue

            if (nf, x, y) == goal:
                continue

            lo, hi = min(f, nf), max(f, nf)
            ok = True
            for g in range(lo, hi + 1):
                if g != f:
                    if kind[g, x, y] != EMPTY:
                        ok = False
                        break

            if ok:
                out.append(((nf, x, y, NO_DIR), LIFT + FLOOR_COST * abs(nf - f)))

    return out

def rebuild(prev, end):
    """Builds a path of cells from the start to end"""
    path = []
    s = end

    while s is not None:
        path.append(s[:3])
        s = prev[s]

    return path[::-1]

def route(kind, start, goal):
    """Cheapest belt from cell start to cell goal: (path, cost), or None if there is no way.

    A*: states (f, x, y, d), heuristic distance(cell, goal). path includes start and goal.
    """
    s0 = (*start, NO_DIR)
    g = {s0: 0}
    prev = {s0: None}
    done = set()
    tie = itertools.count()
    heap = [(distance(start, goal), next(tie), s0)]

    while heap:
        _, _, s = heapq.heappop(heap)
        if s in done: 
            continue

        done.add(s)
        if s[:3] == goal: 
            return rebuild(prev, s), g[s]

        for n, c in neighbours(kind, s, goal):
            ng = g[s] + c
            if ng < g.get(n, float("inf")):
                g[n] = ng
                prev[n] = s
                heapq.heappush(heap, (ng + distance(n[:3], goal), next(tie), n))

    return None