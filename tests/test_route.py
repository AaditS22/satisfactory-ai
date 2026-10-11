import heapq

import numpy as np

from layout.route import (BELT_X, BELT_Y, BLOCKED, CROSS, EMPTY, LIFT, NO_DIR, STEP, TURN,
                          neighbours, rebuild, route)


def grid(F, X, Y):
    return np.full((F, X, Y), EMPTY, dtype=np.int8)


def moves(kind, state, goal=(9, 9, 9)):
    return sorted(neighbours(kind, state, goal))


def check_path(kind, start, goal, path):
    assert path[0] == start and path[-1] == goal
    assert len(set(path)) == len(path)
    total, d = 0, None
    for i in range(1, len(path)):
        (f0, x0, y0), (f1, x1, y1) = path[i - 1], path[i]
        if (x0, y0) == (x1, y1):
            assert f0 != f1 and d is not None
            assert 0 < i < len(path) - 1
            assert kind[f0, x0, y0] == EMPTY
            for f in range(min(f0, f1), max(f0, f1) + 1):
                assert f == f0 or kind[f, x1, y1] == EMPTY
            total += LIFT + 16 * abs(f1 - f0)
            d = None
            continue
        assert f0 == f1 and abs(x1 - x0) + abs(y1 - y0) == 1
        nd = (x1 - x0, y1 - y0)
        k0, k1 = kind[f0, x0, y0], kind[f1, x1, y1]
        if i > 1 and k0 in (BELT_X, BELT_Y):
            assert nd == d
        total += STEP
        if d is not None and nd != d:
            assert nd != (-d[0], -d[1])
            total += TURN
        if path[i] != goal:
            if k1 == BELT_X:
                assert nd[0] == 0
                total += CROSS
            elif k1 == BELT_Y:
                assert nd[1] == 0
                total += CROSS
            else:
                assert k1 == EMPTY
        d = nd
    return total


def dijkstra_cost(kind, start, goal):
    s0 = (*start, NO_DIR)
    best = {s0: 0}
    heap = [(0, s0)]
    while heap:
        c, s = heapq.heappop(heap)
        if c > best[s]:
            continue
        if s[:3] == goal:
            return c
        for n, step in neighbours(kind, s, goal):
            if c + step < best.get(n, float("inf")):
                best[n] = c + step
                heapq.heappush(heap, (c + step, n))
    return None


def test_constants():
    assert (EMPTY, BLOCKED, BELT_X, BELT_Y) == (0, 1, 2, 3)
    assert (STEP, TURN, CROSS, LIFT, NO_DIR) == (1, 2, 6, 8, 4)


def test_neighbours_flat():
    k = grid(1, 5, 5)
    assert moves(k, (0, 2, 2, NO_DIR)) == [((0, 1, 2, 2), 1), ((0, 2, 1, 3), 1), ((0, 2, 3, 1), 1), ((0, 3, 2, 0), 1)]
    assert moves(k, (0, 2, 2, 0)) == [((0, 2, 1, 3), 3), ((0, 2, 3, 1), 3), ((0, 3, 2, 0), 1)]
    assert moves(k, (0, 0, 0, NO_DIR)) == [((0, 0, 1, 1), 1), ((0, 1, 0, 0), 1)]
    assert moves(k, (0, 4, 4, 1)) == [((0, 3, 4, 2), 3)]
    k[0, 3, 2] = BLOCKED
    k[0, 2, 3] = BLOCKED
    assert moves(k, (0, 2, 2, 0)) == [((0, 2, 1, 3), 3)]
    assert moves(k, (0, 2, 2, 0), goal=(0, 3, 2)) == [((0, 2, 1, 3), 3), ((0, 3, 2, 0), 1)]


def test_neighbours_crossing():
    k = grid(1, 5, 5)
    k[0, 3, 2] = BELT_Y
    k[0, 2, 3] = BELT_Y
    assert moves(k, (0, 2, 2, 0)) == [((0, 2, 1, 3), 3), ((0, 3, 2, 0), 7)]
    assert moves(k, (0, 3, 2, 0)) == [((0, 4, 2, 0), 1)]
    k[0, 4, 2] = BELT_X
    assert moves(k, (0, 3, 2, 0)) == []
    k[0, 4, 2] = BELT_Y
    assert moves(k, (0, 3, 2, 0)) == [((0, 4, 2, 0), 7)]
    k = grid(1, 5, 5)
    k[0, 2, 3] = BELT_X
    assert moves(k, (0, 2, 2, 1)) == [((0, 1, 2, 2), 3), ((0, 2, 3, 1), 7), ((0, 3, 2, 0), 3)]


def test_neighbours_lifts():
    k = grid(3, 4, 4)
    assert moves(k, (0, 1, 1, NO_DIR)) == [((0, 0, 1, 2), 1), ((0, 1, 0, 3), 1), ((0, 1, 2, 1), 1), ((0, 2, 1, 0), 1)]
    m = moves(k, (0, 1, 1, 0))
    assert ((1, 1, 1, NO_DIR), 24) in m and ((2, 1, 1, NO_DIR), 40) in m
    m = moves(k, (0, 1, 1, 0), goal=(1, 1, 1))
    assert not any(s[0] == 1 for s, _ in m) and ((2, 1, 1, NO_DIR), 40) in m
    m = moves(k, (2, 1, 1, 3))
    assert ((1, 1, 1, NO_DIR), 24) in m and ((0, 1, 1, NO_DIR), 40) in m
    k[1, 1, 1] = BLOCKED
    m = moves(k, (0, 1, 1, 0))
    assert not any(s[0] != 0 for s, _ in m)
    k = grid(3, 4, 4)
    k[2, 1, 1] = BELT_X
    m = moves(k, (0, 1, 1, 0))
    assert ((1, 1, 1, NO_DIR), 24) in m and not any(s[0] == 2 for s, _ in m)
    k = grid(3, 4, 4)
    k[0, 1, 1] = BELT_Y
    assert not any(s[0] != 0 for s, _ in moves(k, (0, 1, 1, 0)))


def test_rebuild():
    a, b, c = (0, 0, 0, NO_DIR), (0, 1, 0, 0), (1, 1, 0, NO_DIR)
    prev = {a: None, b: a, c: b}
    assert rebuild(prev, c) == [(0, 0, 0), (0, 1, 0), (1, 1, 0)]
    assert rebuild(prev, a) == [(0, 0, 0)]


def test_route_worked_examples():
    k = grid(1, 5, 5)
    path, cost = route(k, (0, 0, 0), (0, 3, 3))
    assert cost == 8 and len(path) == 7
    assert check_path(k, (0, 0, 0), (0, 3, 3), path) == 8
    path, cost = route(k, (0, 0, 2), (0, 4, 2))
    assert path == [(0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 4, 2)] and cost == 4
    k = grid(1, 8, 6)
    k[0, 4, :5] = BLOCKED
    k[0, 1, 1] = BLOCKED
    k[0, 6, 1] = BLOCKED
    path, cost = route(k, (0, 1, 1), (0, 6, 1))
    assert cost == 17 and check_path(k, (0, 1, 1), (0, 6, 1), path) == 17


def test_route_crossing_vs_detour():
    k = grid(1, 7, 9)
    k[0, 3, 0:8] = BELT_Y
    path, cost = route(k, (0, 0, 3), (0, 6, 3))
    assert cost == 12 and (0, 3, 3) in path
    k[0, 3, 0:3] = EMPTY
    path, cost = route(k, (0, 0, 3), (0, 6, 3))
    assert cost == 12
    k[0, 3, 0:5] = EMPTY
    path, cost = route(k, (0, 0, 3), (0, 6, 3))
    assert cost == 6 and check_path(k, (0, 0, 3), (0, 6, 3), path) == 6


def test_route_lift_and_none():
    k = grid(2, 6, 3)
    k[0, 3, :] = BLOCKED
    path, cost = route(k, (0, 0, 1), (0, 5, 1))
    assert cost == 53 and check_path(k, (0, 0, 1), (0, 5, 1), path) == 53
    assert any(p[0] == 1 for p in path)
    k[1, 3, :] = BLOCKED
    assert route(k, (0, 0, 1), (0, 5, 1)) is None
    k = grid(1, 3, 3)
    k[0, 1, :] = BLOCKED
    assert route(k, (0, 0, 1), (0, 2, 1)) is None


def test_route_random():
    rng = np.random.default_rng(0)
    found = 0
    for _ in range(150):
        k = grid(3, 9, 9)
        r = rng.random(k.shape)
        k[r < 0.25] = BLOCKED
        k[(r >= 0.25) & (r < 0.32)] = BELT_X
        k[(r >= 0.32) & (r < 0.39)] = BELT_Y
        cells = [(int(f), int(x), int(y)) for f, x, y in np.argwhere(k == EMPTY)]
        i, j = rng.choice(len(cells), 2, replace=False)
        start, goal = cells[i], cells[j]
        k[start] = BLOCKED
        k[goal] = BLOCKED
        res = route(k, start, goal)
        want = dijkstra_cost(k, start, goal)
        if res is None:
            assert want is None
            continue
        found += 1
        path, cost = res
        assert cost == want
        assert check_path(k, start, goal, path) == cost
    assert found > 50


if __name__ == "__main__":
    tests = [test_constants, test_neighbours_flat, test_neighbours_crossing, test_neighbours_lifts,
             test_rebuild, test_route_worked_examples, test_route_crossing_vs_detour,
             test_route_lift_and_none, test_route_random]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")