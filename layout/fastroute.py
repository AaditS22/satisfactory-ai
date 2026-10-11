import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

from layout.connect import FLOOR_COST
from layout.route import EMPTY, BELT_X, BELT_Y, STEP, TURN, CROSS, LIFT, NO_DIR

INF = np.int64(1) << 62
TIE_BITS = 32
DX = np.array([1, 0, -1, 0], dtype=np.int64)
DY = np.array([0, 1, 0, -1], dtype=np.int64)


@njit(cache=True)
def _push(pri, sta, n, p, s):
    if n == pri.shape[0]:
        pri2 = np.empty(2 * n, dtype=np.int64)
        sta2 = np.empty(2 * n, dtype=np.int64)
        pri2[:n] = pri
        sta2[:n] = sta
        pri = pri2
        sta = sta2
    i = n
    pri[i] = p
    sta[i] = s
    while i > 0:
        parent = (i - 1) // 2
        if pri[parent] <= pri[i]:
            break
        pri[parent], pri[i] = pri[i], pri[parent]
        sta[parent], sta[i] = sta[i], sta[parent]
        i = parent
    return pri, sta, n + 1


@njit(cache=True)
def _pop(pri, sta, n):
    p = pri[0]
    s = sta[0]
    n -= 1
    pri[0] = pri[n]
    sta[0] = sta[n]
    i = 0
    while True:
        a = 2 * i + 1
        b = a + 1
        m = i
        if a < n and pri[a] < pri[m]:
            m = a
        if b < n and pri[b] < pri[m]:
            m = b
        if m == i:
            break
        pri[m], pri[i] = pri[i], pri[m]
        sta[m], sta[i] = sta[i], sta[m]
        i = m
    return p, s, n


@njit(cache=True)
def _astar(kind, extra, sf, sx, sy, gf, gx, gy):
    F, X, Y = kind.shape
    total = F * X * Y * 5
    g = np.full(total, INF, dtype=np.int64)
    prev = np.full(total, -1, dtype=np.int64)
    closed = np.zeros(total, dtype=np.uint8)
    pri = np.empty(1024, dtype=np.int64)
    sta = np.empty(1024, dtype=np.int64)
    n = 0
    tie = 0
    s0 = ((sf * X + sx) * Y + sy) * 5 + NO_DIR
    g[s0] = 0
    h0 = abs(sx - gx) + abs(sy - gy) + FLOOR_COST * abs(sf - gf)
    pri, sta, n = _push(pri, sta, n, (h0 << TIE_BITS) | tie, s0)
    tie += 1
    found = -1
    expanded = 0
    while n > 0:
        p, s, n = _pop(pri, sta, n)
        if closed[s]:
            continue
        closed[s] = 1
        expanded += 1
        d = s % 5
        c = s // 5
        y = c % Y
        c //= Y
        x = c % X
        f = c // X
        if f == gf and x == gx and y == gy:
            found = s
            break
        here = kind[f, x, y]
        gs = g[s]
        for nd in range(4):
            if d != NO_DIR and nd == (d + 2) % 4:
                continue
            if (here == BELT_X or here == BELT_Y) and nd != d:
                continue
            nx = x + DX[nd]
            ny = y + DY[nd]
            if nx < 0 or nx >= X or ny < 0 or ny >= Y:
                continue
            cost = STEP
            if d != NO_DIR and nd != d:
                cost += TURN
            k = kind[f, nx, ny]
            if f == gf and nx == gx and ny == gy:
                pass
            elif k == EMPTY:
                pass
            elif (k == BELT_X and (nd == 1 or nd == 3)) or (k == BELT_Y and (nd == 0 or nd == 2)):
                cost += CROSS
            else:
                continue
            ng = gs + cost + extra[f, nx, ny]
            ns = ((f * X + nx) * Y + ny) * 5 + nd
            if ng < g[ns]:
                g[ns] = ng
                prev[ns] = s
                hn = abs(nx - gx) + abs(ny - gy) + FLOOR_COST * abs(f - gf)
                pri, sta, n = _push(pri, sta, n, ((ng + hn) << TIE_BITS) | tie, ns)
                tie += 1
        if d != NO_DIR and here == EMPTY:
            for nf in range(F):
                if nf == f:
                    continue
                if nf == gf and x == gx and y == gy:
                    continue
                lo = min(f, nf)
                hi = max(f, nf)
                ok = True
                for q in range(lo, hi + 1):
                    if q != f and kind[q, x, y] != EMPTY:
                        ok = False
                        break
                if not ok:
                    continue
                ng = gs + LIFT + FLOOR_COST * abs(nf - f) + extra[nf, x, y]
                ns = ((nf * X + x) * Y + y) * 5 + NO_DIR
                if ng < g[ns]:
                    g[ns] = ng
                    prev[ns] = s
                    hn = abs(x - gx) + abs(y - gy) + FLOOR_COST * abs(nf - gf)
                    pri, sta, n = _push(pri, sta, n, ((ng + hn) << TIE_BITS) | tie, ns)
                    tie += 1
    if found < 0:
        return np.empty((0, 3), dtype=np.int64), np.int64(-1), expanded
    length = 0
    s = found
    while s >= 0:
        length += 1
        s = prev[s]
    path = np.empty((length, 3), dtype=np.int64)
    s = found
    i = length - 1
    while s >= 0:
        c = s // 5
        path[i, 2] = c % Y
        c //= Y
        path[i, 1] = c % X
        path[i, 0] = c // X
        i -= 1
        s = prev[s]
    return path, g[found], expanded


def route_fast(kind, start, goal, extra=None):
    kind = np.ascontiguousarray(kind, dtype=np.int8)
    if extra is None:
        extra = np.zeros(kind.shape, dtype=np.int64)
    else:
        extra = np.ascontiguousarray(extra, dtype=np.int64)
    path, cost, _ = _astar(kind, extra, *(int(v) for v in start), *(int(v) for v in goal))
    if cost < 0:
        return None
    return [tuple(int(v) for v in row) for row in path], int(cost)


def bench(seeds=range(40)):
    """Time route_all with the fast router on finished random episodes (first call compiles, not counted)."""
    import time
    import layout.wire as wire
    from layout.view import random_episode

    k = np.zeros((1, 3, 3), dtype=np.int8)
    route_fast(k, (0, 0, 0), (0, 2, 2))
    rows = []
    for seed in seeds:
        env = random_episode(seed)
        if env.failed:
            continue
        t = time.perf_counter()
        res = wire.route_all(env)
        ms = (time.perf_counter() - t) * 1000
        machines = sum(b["block"]["n"] for b in env.world.blocks)
        rows.append((ms, seed, machines, len(res["paths"]) + len(res["failed"]), res["rounds"]))
    rows.sort()
    print(f"numba: {'yes' if HAVE_NUMBA else 'NO (running as plain Python, slow)'}")
    print(f"{len(rows)} episodes  median {rows[len(rows) // 2][0]:.1f} ms  max {rows[-1][0]:.1f} ms")
    big = [r for r in rows if r[2] >= 25]
    if big:
        print(f"episodes with 25+ machines: {len(big)}, max {max(r[0] for r in big):.1f} ms")
    print("slowest 3 (ms, seed, machines, links, rounds):")
    for r in rows[-3:][::-1]:
        print(f"  {r[0]:7.1f}  seed {r[1]}  {r[2]} machines  {r[3]} links  {r[4]} rounds")


if __name__ == "__main__":
    bench()