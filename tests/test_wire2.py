import random

import numpy as np

from layout.route import route, EMPTY, BLOCKED, BELT_X, BELT_Y
from layout.view import random_episode
from layout.wire import base_kind, links, lay, route_round, blame, route_all, ROUNDS, HIST


def rand_grid(rng):
    k = np.zeros((2, 7, 6), dtype=np.int8)
    for _ in range(12):
        k[rng.randrange(2), rng.randrange(7), rng.randrange(6)] = rng.choice([BLOCKED, BELT_X, BELT_Y])
    return k


def test_route_extra_none_or_zero_same():
    rng = random.Random(1)
    for _ in range(30):
        k = rand_grid(rng)
        s = (0, 0, 0)
        g = (1, 6, 5)
        k[s] = EMPTY
        k[g] = BLOCKED
        a = route(k, s, g)
        assert route(k, s, g, None) == a
        b = route(k, s, g, np.zeros(k.shape, dtype=np.int32))
        assert (a is None) == (b is None)
        if a is not None:
            assert a[1] == b[1]


def test_route_extra_detour():
    k = np.zeros((1, 5, 3), dtype=np.int8)
    extra = np.zeros(k.shape, dtype=np.int32)
    extra[0, 2, 1] = 5
    path, cost = route(k, (0, 0, 1), (0, 4, 1), extra)
    assert cost == 9
    assert (0, 2, 1) in path
    extra[0, 2, 1] = 20
    path, cost = route(k, (0, 0, 1), (0, 4, 1), extra)
    assert cost == 10
    assert (0, 2, 1) not in path
    extra[0, 2, 1] = 0
    extra[0, 0, 1] = 7
    path, cost = route(k, (0, 0, 1), (0, 4, 1), extra)
    assert cost == 4
    assert k.sum() == 0


def test_route_round_episode():
    env = random_episode(0)
    base = base_kind(env)
    keep = base.copy()
    ls = links(env)
    paths, failed, kind, cost = route_round(base, ls, np.zeros(base.shape, dtype=np.int32))
    assert np.array_equal(base, keep)
    assert failed == []
    assert set(paths) == set(ls)
    total = 0
    for L in ls:
        assert paths[L][0] == L[1] and paths[L][-1] == L[2]
        total += route(base, L[1], L[2])[1]
    assert cost >= total
    assert int((kind != base).sum()) > 0


def test_route_round_failure():
    base = np.zeros((1, 5, 3), dtype=np.int8)
    base[0, :, 0] = BLOCKED
    base[0, :, 2] = BLOCKED
    a = ("a", (0, 0, 1), (0, 4, 1), 10.0)
    b = ("b", (0, 1, 1), (0, 3, 1), 5.0)
    base[0, 1, 1] = BLOCKED
    base[0, 3, 1] = BLOCKED
    paths, failed, kind, cost = route_round(base, [b, a], np.zeros(base.shape, dtype=np.int32))
    assert list(paths) == [b]
    assert failed == [a]
    assert cost == 2
    assert kind[0, 2, 1] == BELT_X


def test_blame_hand():
    base = np.zeros((1, 5, 3), dtype=np.int8)
    base[0, 0, 1] = BLOCKED
    base[0, 4, 1] = BLOCKED
    kind = base.copy()
    kind[0, 2, 1] = BLOCKED
    kind[0, 2, 0] = BELT_Y
    hist = np.zeros(base.shape, dtype=np.int32)
    L = ("x", (0, 0, 1), (0, 4, 1), 1.0)
    assert blame(base, kind, [L], hist, 4) is None
    assert hist[0, 2, 1] == 4
    assert hist.sum() == 4
    blame(base, kind, [L, L], hist, 3)
    assert hist[0, 2, 1] == 10
    assert hist.sum() == 10


def check_layout(env, res):
    base = base_kind(env)
    ls = links(env)
    assert len(res["paths"]) + len(res["failed"]) == len(ls)
    assert set(res["paths"]) | set(res["failed"]) == set(ls)
    assert 1 <= res["rounds"] <= ROUNDS
    use = {}
    total = 0
    for L, path in res["paths"].items():
        assert path[0] == L[1] and path[-1] == L[2]
        for i in range(1, len(path) - 1):
            p, c, n = path[i - 1], path[i], path[i + 1]
            assert base[c] == EMPTY
            if p[0] != c[0] or n[0] != c[0]:
                lo, hi = min(p[0], c[0], n[0]), max(p[0], c[0], n[0])
                for f in range(lo, hi + 1):
                    use.setdefault((f, c[1], c[2]), []).append((L, "lift"))
            else:
                a = (c[1] - p[1], c[2] - p[2])
                b = (n[1] - c[1], n[2] - c[2])
                t = "turn" if a != b else ("x" if a[1] == 0 else "y")
                use.setdefault(c, []).append((L, t))
    for c, us in use.items():
        owners = {u[0] for u in us}
        if len(owners) == 1:
            continue
        assert len(owners) == 2 and len(us) == 2, c
        assert sorted(u[1] for u in us) == ["x", "y"], c
    for L, path in res["paths"].items():
        k = base.copy()
        r = route(k, L[1], L[2])
        assert r is not None


def test_route_all_seed4():
    env = random_episode(4)
    res = route_all(env)
    assert res["rounds"] == 2
    assert res["failed"] == []
    assert res["cost"] == 470
    assert len(res["paths"]) == len(links(env)) == 9
    check_layout(env, res)


def test_route_all_valid_and_repeatable():
    for seed in (0, 2, 4):
        env = random_episode(seed)
        res = route_all(env)
        check_layout(env, res)
        again = route_all(env)
        assert again["paths"] == res["paths"]
        assert again["cost"] == res["cost"]
        k = base_kind(env)
        for path in res["paths"].values():
            lay(k, path)
        assert np.array_equal(k, res["kind"])


def test_route_all_reports_failure():
    env = random_episode(26)
    res = route_all(env)
    assert res["rounds"] == ROUNDS
    assert len(res["failed"]) == 1
    L = res["failed"][0]
    assert L in links(env)
    assert L not in res["paths"]
    check_layout(env, res)
    assert route_all(env, rounds=1)["rounds"] == 1


TESTS = [
    test_route_extra_none_or_zero_same,
    test_route_extra_detour,
    test_route_round_episode,
    test_route_round_failure,
    test_blame_hand,
    test_route_all_seed4,
    test_route_all_valid_and_repeatable,
    test_route_all_reports_failure,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")