import numpy as np

from layout.env import LayoutEnv
from layout.world import World, EMPTY, end_cells
from layout.connect import ends_for, distance
from layout.view import random_episode
from layout.baseline import corridor_mask, wire_cost, W_OUT, best_move, greedy, baseline_a
from layout.api import play, evaluate


class Tiny:
    pass


def tiny_env():
    env = Tiny()
    env.world = World(1, 6, 4)
    env.world.owner[0, 0:2, 0:2] = 0
    return env


def picture(rows):
    return np.array(rows[::-1], dtype=bool).T[None]


def slow_corridor(env, w, d, g, axis):
    F, X, Y = env.world.owner.shape
    out = np.zeros((F, X - w + 1, Y - d + 1), dtype=bool)
    for f in range(F):
        for x in range(X - w + 1):
            for y in range(Y - d + 1):
                ok = True
                for cx in range(x - g, x + w + g):
                    for cy in range(y - g, y + d + g):
                        if 0 <= cx < X and 0 <= cy < Y:
                            if env.world.owner[f, cx, cy] != EMPTY:
                                ok = False
                        elif axis == 0 and not 0 <= cx < X:
                            ok = False
                        elif axis == 1 and not 0 <= cy < Y:
                            ok = False
                out[f, x, y] = ok
    return out


def mid_env(seed, k):
    rng = np.random.default_rng(1000 + seed)
    from layout.tasks import random_task
    env = LayoutEnv()
    env.reset(random_task(rng))
    while not env.done and len(env.world.blocks) < k:
        ns = np.flatnonzero(env.n_mask())
        n = int(ns[rng.integers(len(ns))])
        spots = np.argwhere(env.pos_mask(n))
        rot, f, x, y = (int(v) for v in spots[rng.integers(len(spots))])
        env.step(n, rot, f, x, y)
    return env, rng


def plate_task(floors=2, side=0, rate=30, width=34, depth=48):
    return {"target": "Desc_IronPlate_C", "rate": rate, "width": width, "depth": depth,
            "floors": floors, "side": side, "alternates": set()}


def plate_env():
    task = plate_task()
    env = LayoutEnv()
    env.reset(task)
    env.step(2, 0, 0, 2, 2)
    return env


def test_corridor_hand_open():
    got = corridor_mask(tiny_env(), 1, 1, 1, None)
    want = picture([[1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1]])
    assert got.shape == (1, 6, 4)
    assert got.dtype == bool
    assert (got == want).all()


def test_corridor_hand_walls_x():
    got = corridor_mask(tiny_env(), 1, 1, 1, 0)
    want = picture([[0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0]])
    assert (got == want).all()


def test_corridor_hand_walls_y():
    got = corridor_mask(tiny_env(), 1, 1, 1, 1)
    want = picture([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0]])
    assert (got == want).all()


def test_corridor_bigger_block():
    env = tiny_env()
    got = corridor_mask(env, 2, 3, 0, None)
    assert got.shape == (1, 5, 2)
    want = np.zeros((1, 5, 2), dtype=bool)
    want[0, 2:, :] = True
    assert (got == want).all()
    wide = corridor_mask(env, 2, 1, 2, None)
    assert wide.shape == (1, 5, 4)
    assert wide.sum() == 4 and wide[0, 4].all()


def test_corridor_ignores_reserved():
    env = tiny_env()
    env.world.reserved[0, 4, 3] = True
    assert corridor_mask(env, 1, 1, 0, None)[0, 4, 3]


def test_corridor_random_slow():
    rng = np.random.default_rng(7)
    for seed in range(4):
        env = random_episode(seed)
        F, X, Y = env.world.owner.shape
        for _ in range(2):
            w = int(rng.integers(1, min(X, 9)))
            d = int(rng.integers(1, min(Y, 9)))
            g = int(rng.integers(0, 4))
            axis = [None, 0, 1][int(rng.integers(3))]
            before = env.world.owner.copy()
            got = corridor_mask(env, w, d, g, axis)
            assert got.shape == (F, X - w + 1, Y - d + 1)
            assert (got == slow_corridor(env, w, d, g, axis)).all()
            assert (env.world.owner == before).all()


def test_wire_cost_plate_hand():
    env = plate_env()
    blk = env.block(2, 0)
    got = wire_cost(env, blk, np.array([0, 1, 0]), np.array([14, 14, 20]), np.array([17, 17, 30]))
    assert got.shape == (3,)
    assert got.dtype == np.float64
    assert np.allclose(got, [4.3, 25.1, 28.4])


def test_wire_cost_one_step_east():
    env = plate_env()
    blk = env.block(2, 0)
    a = wire_cost(env, blk, np.array([0]), np.array([14]), np.array([17]))[0]
    b = wire_cost(env, blk, np.array([0]), np.array([15]), np.array([17]))[0]
    assert abs((b - a) - (1 - W_OUT)) < 1e-9


def test_wire_cost_random_slow():
    checked = 0
    for seed in range(20):
        env, rng = mid_env(seed, 1 + seed % 2)
        if env.done:
            continue
        n = int(np.flatnonzero(env.n_mask())[0])
        rot = int(rng.integers(4))
        m = env.pos_mask(n)[rot]
        if not m.any():
            continue
        blk = env.block(n, rot)
        f, x, y = np.nonzero(m)
        pick = rng.choice(len(f), size=min(len(f), 25), replace=False)
        f, x, y = f[pick], x[pick], y[pick]
        ends = end_cells(blk)
        tx, ty = env.exit
        want = []
        for i in range(len(f)):
            c = 0.0
            for k, inp in enumerate(blk["inputs"]):
                cell = (f[i], x[i] + ends[k][0], y[i] + ends[k][1])
                c += min(distance(cell, p) for p in ends_for(env, inp["item"])[0])
            out = (f[i], x[i] + ends[-1][0], y[i] + ends[-1][1])
            c += W_OUT * distance(out, (0, tx, ty))
            want.append(c)
        q = len(env.queue)
        got = wire_cost(env, blk, f, x, y)
        assert np.allclose(got, want)
        assert len(env.queue) == q
        checked += 1
    assert checked >= 6


def test_best_move_plate():
    env = plate_env()
    n, cost, move = best_move(env, 0, 0)
    assert n == 2
    assert move == (2, 0, 0, 15, 17)
    assert abs(cost - 13.0) < 1e-9
    assert env.pos_mask(2)[move[1:]]
    assert len(env.world.blocks) == 1


def test_best_move_none_and_fallback():
    task = plate_task(floors=1, side=1, rate=15, width=8, depth=48)
    env = LayoutEnv()
    env.reset(task)
    assert best_move(env, 1, 1) is None
    assert best_move(env, 3, 1) is None
    assert best_move(env, 0, 1) == (1, 17.0, (1, 0, 0, 2, 3))
    plan = greedy(task)
    assert plan == [[1, 0, 0, 2, 3]]
    assert abs(evaluate(task, plan)["score"] + 1.5) < 1e-9


def test_greedy_plate_plans():
    task = plate_task(floors=1)
    assert greedy(task, False) == [[2, 0, 0, 3, 22], [2, 0, 0, 14, 28]]
    assert greedy(task, True) == [[2, 3, 0, 3, 14], [2, 1, 0, 3, 27]]
    assert greedy(plate_task(side=1), False) == [[2, 1, 0, 0, 3], [2, 1, 0, 0, 16]]


def test_baseline_a_plate():
    task = plate_task(floors=1)
    res = baseline_a(task)
    assert res["plan"] == [[2, 0, 0, 3, 22], [2, 0, 0, 14, 28]]
    assert abs(res["score"] - 0.7943544097308415) < 1e-9
    assert res["terms"]["passed"]
    assert set(res) >= {"env", "routes", "score", "terms", "plan"}


def test_baseline_a_keeps_best_random():
    from layout.tasks import random_task
    for seed in (0, 2, 3, 4, 6, 7):
        task = random_task(np.random.default_rng(seed))
        res = baseline_a(task)
        a = evaluate(task, greedy(task, False))["score"]
        b = evaluate(task, greedy(task, True))["score"]
        assert abs(res["score"] - max(a, b)) < 1e-12
        env = play(task, res["plan"])
        assert env.done
        assert abs(evaluate(task, res["plan"])["score"] - res["score"]) < 1e-12


def test_greedy_free_takes_largest_n():
    from layout.tasks import random_task
    for seed in (0, 13, 27):
        task = random_task(np.random.default_rng(seed))
        plan = greedy(task, True)
        env = LayoutEnv()
        env.reset(task)
        for move in plan:
            ns = [m[0] for m in (best_move(env, r, task["side"] % 2) for r in range(4)) if m is not None]
            assert move[0] == max(ns)
            env.step(*move)


TESTS = [test_corridor_hand_open, test_corridor_hand_walls_x, test_corridor_hand_walls_y,
         test_corridor_bigger_block, test_corridor_ignores_reserved, test_corridor_random_slow,
         test_wire_cost_plate_hand, test_wire_cost_one_step_east, test_wire_cost_random_slow,
         test_best_move_plate, test_best_move_none_and_fallback, test_greedy_plate_plans, test_greedy_free_takes_largest_n,
         test_baseline_a_plate, test_baseline_a_keeps_best_random]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(TESTS)} passed")