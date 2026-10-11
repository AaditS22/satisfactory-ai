import random

import numpy as np

import layout.wire as wire
from layout.route import route, EMPTY, BLOCKED, BELT_X, BELT_Y
from layout.fastroute import route_fast
from layout.view import random_episode


def rand_case(rng):
    F, X, Y = rng.choice([1, 2, 3]), rng.randint(2, 9), rng.randint(2, 9)
    k = np.zeros((F, X, Y), dtype=np.int8)
    for _ in range(rng.randint(0, F * X * Y // 2)):
        k[rng.randrange(F), rng.randrange(X), rng.randrange(Y)] = rng.choice([BLOCKED, BELT_X, BELT_Y])
    s = (rng.randrange(F), rng.randrange(X), rng.randrange(Y))
    g = (rng.randrange(F), rng.randrange(X), rng.randrange(Y))
    k[s] = BLOCKED
    k[g] = rng.choice([BLOCKED, EMPTY])
    e = np.array([[[rng.choice([0, 0, 0, 3, 9]) for _ in range(Y)] for _ in range(X)] for _ in range(F)], dtype=np.int32)
    return k, s, g, e


def test_hand_straight():
    k = np.zeros((1, 5, 3), dtype=np.int8)
    assert route_fast(k, (0, 0, 1), (0, 4, 1)) == ([(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)], 4)
    e = np.zeros(k.shape, dtype=np.int32)
    e[0, 2, 1] = 20
    path, cost = route_fast(k, (0, 0, 1), (0, 4, 1), e)
    assert cost == 10 and (0, 2, 1) not in path


def test_types():
    k = np.zeros((2, 4, 4), dtype=np.int8)
    path, cost = route_fast(k, (0, 0, 0), (1, 3, 3))
    assert type(cost) is int
    assert all(type(c) is tuple and len(c) == 3 and all(type(v) is int for v in c) for c in path)
    assert path[0] == (0, 0, 0) and path[-1] == (1, 3, 3)
    assert k.sum() == 0


def test_never_lifts_onto_goal():
    k = np.zeros((2, 3, 3), dtype=np.int8)
    path, cost = route_fast(k, (0, 0, 1), (1, 1, 1))
    assert path == route(k, (0, 0, 1), (1, 1, 1))[0]
    for a, b in zip(path, path[1:]):
        if a[0] != b[0]:
            assert b != (1, 1, 1)


def test_none_when_walled():
    k = np.zeros((1, 5, 5), dtype=np.int8)
    k[0, 2, :] = BLOCKED
    assert route_fast(k, (0, 0, 2), (0, 4, 2)) is None
    k[0, 2, 2] = BELT_X
    assert route_fast(k, (0, 0, 2), (0, 4, 2)) is None
    k[0, 2, 2] = BELT_Y
    assert route_fast(k, (0, 0, 2), (0, 4, 2)) == route(k, (0, 0, 2), (0, 4, 2))


def test_matches_route_random():
    rng = random.Random(7)
    n = 0
    for _ in range(250):
        k, s, g, e = rand_case(rng)
        if s == g:
            continue
        assert route_fast(k, s, g) == route(k, s, g)
        assert route_fast(k, s, g, e) == route(k, s, g, e)
        n += 1
    assert n > 200


def test_route_all_same_as_slow():
    fast = wire.route
    try:
        for seed in (4, 26):
            env = random_episode(seed)
            wire.route = route
            a = wire.route_all(env)
            wire.route = route_fast
            b = wire.route_all(env)
            assert a["paths"] == b["paths"]
            assert a["failed"] == b["failed"]
            assert a["cost"] == b["cost"] and a["rounds"] == b["rounds"]
            assert np.array_equal(a["kind"], b["kind"])
    finally:
        wire.route = fast


def test_wire_uses_fast():
    assert wire.route is route_fast


TESTS = [
    test_hand_straight,
    test_types,
    test_never_lifts_onto_goal,
    test_none_when_walled,
    test_matches_route_random,
    test_route_all_same_as_slow,
    test_wire_uses_fast,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")