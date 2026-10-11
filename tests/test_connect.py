import numpy as np

from layout.connect import FLOOR_COST, chain_order, chains, distance, ends_for, nearest, trunk_rates
from layout.env import LayoutEnv
from layout.view import random_episode


def slow_nearest(cell, cells, used):
    left = [(distance(cell, c), i) for i, c in enumerate(cells) if i not in used]
    return min(left)[1] if left else None


def random_cells(rng, k):
    return [(int(rng.integers(0, 3)), int(rng.integers(0, 40)), int(rng.integers(0, 40))) for _ in range(k)]


def test_distance_worked_examples():
    assert FLOOR_COST == 16
    assert distance((0, 2, 3), (1, 5, 1)) == 21
    assert distance((1, 5, 1), (0, 2, 3)) == 21
    assert distance((0, 5, 5), (0, 1, 9)) == 8
    assert distance((2, 0, 0), (0, 0, 0)) == 32
    assert distance((0, 0, 0), (2, 0, 0)) == 32
    assert distance((1, 7, 4), (1, 7, 4)) == 0
    assert distance((0, 0, 0), (0, 95, 95)) == 190


def test_distance_invariants():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b, c = random_cells(rng, 3)
        assert distance(a, b) == distance(b, a)
        assert distance(a, b) >= 0
        assert distance(a, c) <= distance(a, b) + distance(b, c)
        assert (distance(a, b) == 0) == (a == b)


def test_nearest_worked_examples():
    cells = [(0, 3, 0), (0, 0, 2), (1, 0, 0), (0, 2, 0)]
    assert nearest((0, 0, 0), cells, set()) == 1
    assert nearest((0, 0, 0), cells, {1}) == 3
    assert nearest((0, 0, 0), cells, {1, 3}) == 0
    assert nearest((0, 0, 0), cells, {0, 1, 3}) == 2
    assert nearest((0, 0, 0), cells, {0, 1, 2, 3}) is None
    assert nearest((1, 0, 0), cells, set()) == 2
    assert nearest((0, 9, 9), [(0, 9, 9)], set()) == 0


def test_nearest_does_not_change_inputs():
    cells = [(0, 3, 0), (0, 0, 2), (1, 0, 0)]
    used = {0}
    nearest((0, 0, 0), cells, used)
    assert used == {0}
    assert cells == [(0, 3, 0), (0, 0, 2), (1, 0, 0)]


def test_nearest_random():
    rng = np.random.default_rng(1)
    for _ in range(300):
        k = int(rng.integers(1, 9))
        cells = random_cells(rng, k)
        cell = random_cells(rng, 1)[0]
        used = {i for i in range(k) if rng.random() < 0.4}
        assert nearest(cell, cells, used) == slow_nearest(cell, cells, used)


def test_chain_order_worked_example():
    prods = [(0, 0, 2), (0, 2, 10), (1, 4, 6)]
    cons = [(0, 20, 4), (0, 20, 12), (1, 22, 8)]
    assert chain_order(prods, cons) == ([0, 1, 2], [2, 0, 1])


def test_chain_order_single():
    assert chain_order([(0, 3, 3)], [(2, 50, 7)]) == ([0], [0])


def test_chain_order_start():
    assert chain_order([(0, 10, 0), (0, 0, 0)], [(0, 12, 0)]) == ([1, 0], [0])
    assert chain_order([(0, 0, 0), (0, 10, 0)], [(0, 5, 0)]) == ([0, 1], [0])
    assert chain_order([(0, 0, 0), (0, 10, 0)], [(0, 1, 0), (0, 30, 0)]) == ([1, 0], [0, 1])


def test_chain_order_consumers_start_from_last_producer():
    prods = [(0, 0, 0), (0, 0, 20)]
    cons = [(0, 1, 0), (0, 1, 20)]
    assert chain_order(prods, cons) == ([0, 1], [1, 0])


def test_chain_order_does_not_change_inputs():
    prods = [(0, 0, 2), (0, 2, 10), (1, 4, 6)]
    cons = [(0, 20, 4), (0, 20, 12), (1, 22, 8)]
    chain_order(prods, cons)
    assert prods == [(0, 0, 2), (0, 2, 10), (1, 4, 6)]
    assert cons == [(0, 20, 4), (0, 20, 12), (1, 22, 8)]


def test_chain_order_random():
    rng = np.random.default_rng(2)
    for _ in range(300):
        prods = random_cells(rng, int(rng.integers(1, 7)))
        cons = random_cells(rng, int(rng.integers(1, 7)))
        p, c = chain_order(prods, cons)
        assert type(p) is list and type(c) is list
        assert sorted(p) == list(range(len(prods)))
        assert sorted(c) == list(range(len(cons)))
        far = [min(distance(q, k) for k in cons) for q in prods]
        assert p[0] == far.index(max(far))
        for i in range(1, len(p)):
            assert p[i] == slow_nearest(prods[p[i - 1]], prods, set(p[:i]))
        assert c[0] == slow_nearest(prods[p[-1]], cons, set())
        for j in range(1, len(c)):
            assert c[j] == slow_nearest(cons[c[j - 1]], cons, set(c[:j]))


def close(a, b):
    return len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))


def hand_env(steps=True):
    env = LayoutEnv()
    env.reset({"target": "Desc_SpikedRebar_C", "rate": 20, "width": 48, "depth": 48,
               "floors": 3, "side": 0, "alternates": set()})
    if steps:
        for f, x, y in [(0, 1, 0), (1, 1, 0), (0, 1, 20), (1, 1, 20), (0, 7, 0)]:
            env.step(1, 0, f, x, y)
    return env


def test_trunk_rates():
    assert close(trunk_rates([10, 10, 10], [7.5] * 4, [0, 1, 2], [0, 1, 2, 3]), [10, 20, 30, 22.5, 15, 7.5])
    assert close(trunk_rates([5, 20], [10, 15], [1, 0], [1, 0]), [20, 25, 10])
    assert close(trunk_rates([8], [8], [0], [0]), [8])
    assert close(trunk_rates([4], [1, 3], [0], [1, 0]), [4, 1])


def test_ends_for_hand():
    env = hand_env()
    p, pr, c, cr = ends_for(env, "Desc_IronRod_C")
    assert p == [(1, 10, 17), (0, 10, 37)] and c == [(1, 0, 22), (0, 6, 2)]
    assert close(pr, [10, 10]) and close(cr, [10, 10])
    p, pr, c, cr = ends_for(env, "Desc_OreIron_C")
    assert p == [(0, 0, 24)] and c == [(0, 0, 2)]
    assert close(pr, [20]) and close(cr, [20])
    p, pr, c, cr = ends_for(env, "Desc_SpikedRebar_C")
    assert p == [(1, 10, 37), (0, 16, 17)] and c == [(0, 47, 24)]
    assert close(pr, [10, 10]) and close(cr, [20])


def test_chains_hand():
    out = chains(hand_env())
    assert set(out) == {"Desc_OreIron_C", "Desc_IronIngot_C", "Desc_IronRod_C", "Desc_SpikedRebar_C"}
    expect = {
        "Desc_OreIron_C": ([(0, 0, 24), (0, 0, 2)], [20], 1),
        "Desc_IronIngot_C": ([(0, 6, 17), (0, 0, 22), (1, 0, 2)], [20, 10], 1),
        "Desc_IronRod_C": ([(0, 10, 37), (1, 10, 17), (1, 0, 22), (0, 6, 2)], [10, 20, 10], 2),
        "Desc_SpikedRebar_C": ([(1, 10, 37), (0, 16, 17), (0, 47, 24)], [10, 20], 2),
    }
    for item, (cells, rates, n_prod) in expect.items():
        assert out[item]["cells"] == cells, item
        assert close(out[item]["rates"], rates), item
        assert out[item]["n_prod"] == n_prod, item


def test_chains_unfinished_raises():
    try:
        chains(hand_env(steps=False))
    except ValueError:
        return
    assert False


def test_chains_random_episodes():
    seen = 0
    for seed in range(40):
        env = random_episode(seed)
        if env.failed:
            continue
        seen += 1
        out = chains(env)
        assert set(out) == {e["item"] for e in env.graph["edges"]}
        want = sorted([(b["floor"], *e) for b in env.world.blocks for e in b["ends"]]
                      + [(0, *xy) for xy in env.entries.values()] + [(0, *env.exit)])
        got = sorted(cell for ch in out.values() for cell in ch["cells"])
        assert got == want
        for ch in out.values():
            k, rates = ch["n_prod"], ch["rates"]
            assert 1 <= k < len(ch["cells"])
            assert len(rates) == len(ch["cells"]) - 1
            assert min(rates) > 1e-9
            assert abs(rates[k - 1] - max(rates)) < 1e-6
            assert max(rates) <= 1200 + 1e-9
    assert seen >= 20


if __name__ == "__main__":
    tests = [test_distance_worked_examples, test_distance_invariants,
             test_nearest_worked_examples, test_nearest_does_not_change_inputs, test_nearest_random,
             test_chain_order_worked_example, test_chain_order_single, test_chain_order_start,
             test_chain_order_consumers_start_from_last_producer, test_chain_order_does_not_change_inputs,
             test_chain_order_random, test_trunk_rates, test_ends_for_hand, test_chains_hand,
             test_chains_unfinished_raises, test_chains_random_episodes]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")