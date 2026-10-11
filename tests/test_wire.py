from types import SimpleNamespace

import numpy as np

from layout.connect import chains, distance
from layout.route import route, EMPTY, BLOCKED, BELT_X, BELT_Y
from layout.view import random_episode
from layout.wire import base_kind, links, lay


def fake_env():
    owner = np.full((2, 4, 3), -1, dtype=np.int32)
    owner[0, 1, 0] = 0
    owner[0, 2, 0] = 0
    owner[1, 3, 2] = 1
    reserved = np.zeros((2, 4, 3), dtype=bool)
    reserved[0, 0, 0] = True
    reserved[1, 0, 2] = True
    return SimpleNamespace(world=SimpleNamespace(owner=owner, reserved=reserved))


def test_base_kind_hand():
    env = fake_env()
    k = base_kind(env)
    assert isinstance(k, np.ndarray)
    assert k.dtype == np.int8
    assert k.shape == (2, 4, 3)
    want = {(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 3, 2), (1, 0, 2)}
    for f in range(2):
        for x in range(4):
            for y in range(3):
                assert k[f, x, y] == (BLOCKED if (f, x, y) in want else EMPTY), (f, x, y)
    k[0, 3, 1] = BLOCKED
    assert base_kind(env)[0, 3, 1] == EMPTY


def test_base_kind_episode():
    for seed in (0, 3, 7):
        env = random_episode(seed)
        k = base_kind(env)
        w = env.world
        assert k.shape == w.owner.shape
        assert np.array_equal(k == BLOCKED, (w.owner != -1) | w.reserved)
        assert set(np.unique(k)) <= {EMPTY, BLOCKED}
        for x, y in env.entries.values():
            assert k[0, x, y] == BLOCKED
        assert k[0, env.exit[0], env.exit[1]] == BLOCKED


def test_links_episode():
    env = random_episode(0)
    ls = links(env)
    ch = chains(env)
    assert len(ls) == sum(len(c["cells"]) - 1 for c in ch.values())
    items = [L[0] for L in ls]
    assert items == sorted(items)
    for item, c in ch.items():
        mine = [L for L in ls if L[0] == item]
        assert [L[1] for L in mine] == c["cells"][:-1]
        assert [L[2] for L in mine] == c["cells"][1:]
        assert [L[3] for L in mine] == c["rates"]
    for L in ls:
        assert len(L) == 4
        assert len(L[1]) == 3 and len(L[2]) == 3


def test_links_counts():
    for seed in range(12):
        env = random_episode(seed)
        if env.failed:
            continue
        ls = links(env)
        assert len(ls) == sum(len(c["cells"]) - 1 for c in chains(env).values())
        assert all(L[1] != L[2] for L in ls)


def test_lay_straight_x():
    k = np.zeros((1, 6, 3), dtype=np.int8)
    path = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1)]
    assert lay(k, path) is None
    assert k[0, 1, 1] == BELT_X and k[0, 2, 1] == BELT_X
    assert k[0, 0, 1] == EMPTY and k[0, 3, 1] == EMPTY
    assert int((k != EMPTY).sum()) == 2


def test_lay_turn_and_y():
    k = np.zeros((1, 4, 5), dtype=np.int8)
    path = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
    lay(k, path)
    assert k[0, 1, 0] == BELT_X
    assert k[0, 2, 0] == BLOCKED
    assert k[0, 2, 1] == BELT_Y and k[0, 2, 2] == BELT_Y
    assert k[0, 0, 0] == EMPTY and k[0, 2, 3] == EMPTY
    assert int((k != EMPTY).sum()) == 4


def test_lay_crossing():
    k = np.zeros((1, 5, 5), dtype=np.int8)
    k[0, 2, 2] = BELT_X
    k[0, 1, 2] = BELT_X
    path = [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4)]
    lay(k, path)
    assert k[0, 2, 2] == BLOCKED
    assert k[0, 2, 1] == BELT_Y and k[0, 2, 3] == BELT_Y
    assert k[0, 1, 2] == BELT_X


def test_lay_lift():
    k = np.zeros((3, 5, 4), dtype=np.int8)
    path = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 3)]
    lay(k, path)
    assert k[0, 1, 1] == BELT_X
    assert all(k[f, 2, 1] == BLOCKED for f in range(3))
    assert k[2, 2, 2] == BELT_Y
    assert int((k != EMPTY).sum()) == 5
    k = np.zeros((3, 5, 4), dtype=np.int8)
    path = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 2), (1, 1, 3)]
    lay(k, path)
    assert k[0, 1, 1] == BLOCKED and k[1, 1, 1] == BLOCKED
    assert k[2, 1, 1] == EMPTY
    assert k[1, 1, 2] == BELT_Y
    assert int((k != EMPTY).sum()) == 3


def slow_class(before, p, c, n):
    if p[0] != c[0] or n[0] != c[0]:
        return "lift"
    if before != EMPTY:
        return BLOCKED
    a = (c[1] - p[1], c[2] - p[2])
    b = (n[1] - c[1], n[2] - c[2])
    if a != b:
        return BLOCKED
    return BELT_X if a[1] == 0 else BELT_Y


def test_lay_random_routes():
    for seed in (2, 5, 9):
        env = random_episode(seed)
        if env.failed:
            continue
        k = base_kind(env)
        ls = sorted(links(env), key=lambda L: distance(L[1], L[2]))
        for item, s, g, rate in ls:
            r = route(k, s, g)
            if r is None:
                continue
            path = r[0]
            before = k.copy()
            lay(k, path)
            changed = set(zip(*np.nonzero(k != before)))
            expect = set()
            for i in range(1, len(path) - 1):
                p, c, n = path[i - 1], path[i], path[i + 1]
                want = slow_class(before[c], p, c, n)
                if want == "lift":
                    lo = min(p[0], c[0], n[0])
                    hi = max(p[0], c[0], n[0])
                    for f in range(lo, hi + 1):
                        assert k[f, c[1], c[2]] == BLOCKED
                        if before[f, c[1], c[2]] != BLOCKED:
                            expect.add((f, c[1], c[2]))
                else:
                    assert k[c] == want, (seed, c)
                    if before[c] != want:
                        expect.add(c)
            assert {tuple(int(v) for v in t) for t in changed} == expect
            assert k[s] == before[s] and k[g] == before[g]


TESTS = [
    test_base_kind_hand,
    test_base_kind_episode,
    test_links_episode,
    test_links_counts,
    test_lay_straight_x,
    test_lay_turn_and_y,
    test_lay_crossing,
    test_lay_lift,
    test_lay_random_routes,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")