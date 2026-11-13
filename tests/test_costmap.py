import json

import numpy as np

from layout.costmap import port_cost, square_order, block_metres, snap, block_cost, COARSE, COST_SCALE, COST_CAP
from layout.connect import ends_for
from layout.env import LayoutEnv
from layout.gym_env import offsets, snap_mask, fine_cell


def test_shape_and_dtype():
    m = port_cost(0, 0, [(1, 1)], 5, 4)
    assert m.shape == (5, 4)
    assert m.dtype == np.float32


def test_hand_example():
    m = port_cost(1, 0, [(2, 1)], 3, 2)
    assert np.array_equal(m, np.array([[2, 1], [1, 0], [2, 1]], dtype=np.float32))


def test_offset_moves_the_zero():
    m = port_cost(3, 2, [(5, 5)], 10, 10)
    assert m[2, 3] == 0
    assert m[5, 5] == 5


def test_nearest_of_two_sources():
    m = port_cost(0, 0, [(0, 0), (9, 0)], 10, 1)
    assert m[0, 0] == 0
    assert m[9, 0] == 0
    assert m[4, 0] == 4
    assert m[5, 0] == 4


def test_x_is_rows_y_is_columns():
    m = port_cost(0, 0, [(7, 0)], 8, 3)
    assert m[7, 0] == 0
    assert m[0, 0] == 7
    assert m[7, 2] == 2


def test_no_sources_gives_zeros():
    m = port_cost(2, 2, [], 4, 4)
    assert np.array_equal(m, np.zeros((4, 4), dtype=np.float32))


def test_matches_plain_loop():
    rng = np.random.default_rng(0)
    srcs = [tuple(int(v) for v in rng.integers(0, 20, 2)) for _ in range(3)]
    m = port_cost(4, 1, srcs, 20, 15)
    for x in range(20):
        for y in range(15):
            want = min(abs(x + 4 - sx) + abs(y + 1 - sy) for sx, sy in srcs)
            assert m[x, y] == want


def stage3_env(seed, steps):
    row = [r for r in json.load(open("data/eval_set.json")) if r["seed"] == seed][0]
    task = dict(row["task"])
    task["alternates"] = frozenset(task["alternates"])
    env = LayoutEnv()
    env.reset(task)
    plan = row["A"]["plan"]
    for move in plan[:steps]:
        env.step(*move)
    return env, plan


def test_square_order_matches_gym_env():
    assert square_order() == offsets()


def test_snap_hand_example():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    legal = np.zeros((4, 4), dtype=bool)
    legal[1, 1] = True
    legal[0, 1] = True
    legal[3, 3] = True
    out = snap(values, legal)
    assert out.shape == (COARSE, COARSE)
    assert out[0, 0] == values[0, 1]
    assert out[1, 1] == values[3, 3]
    assert out[0, 1] == 0
    assert out[1, 0] == 0
    assert out[2:, :].sum() == 0


def test_snap_matches_fine_cell():
    env, plan = stage3_env(55007, 3)
    n = plan[3][0]
    fine = env.pos_mask(n)
    legal = snap_mask(fine)
    for rot in range(4):
        metres = block_metres(env, env.block(n, rot))
        for f in range(fine.shape[1]):
            out = snap(metres, fine[rot, f])
            for cx, cy in zip(*np.nonzero(legal[rot, f])):
                x, y = fine_cell(fine, rot, f, cx, cy)
                assert out[cx, cy] == metres[x, y]
            assert (out[:legal.shape[2], :legal.shape[3]][~legal[rot, f]] == 0).all()


def test_block_metres_matches_placed_block():
    for seed in (55000, 55007, 56010):
        env, plan = stage3_env(seed, 0)
        for n, rot, f, x, y in plan:
            want_map = block_metres(env, env.block(n, rot))
            env.step(n, rot, f, x, y)
            b = env.world.blocks[-1]
            got = 0
            for i, inp in enumerate(b["block"]["inputs"]):
                prods = ends_for(env, inp["item"])[0]
                ex, ey = b["ends"][i]
                got += min(abs(ex - px) + abs(ey - py) for _, px, py in prods) if prods else 0
            cons = ends_for(env, b["block"]["output"]["item"])[2]
            ox, oy = b["ends"][-1]
            got += min(abs(ox - cx) + abs(oy - cy) for _, cx, cy in cons) if cons else 0
            assert want_map[x, y] == got
            if env.done:
                break


def test_block_cost_shape_range_and_zeros():
    env, plan = stage3_env(55007, 2)
    n = plan[2][0]
    fine = env.pos_mask(n)
    c = block_cost(env, n, fine)
    assert c.shape == (4, fine.shape[1], COARSE, COARSE)
    assert c.dtype == np.float32
    assert c.min() >= 0 and c.max() <= COST_CAP
    legal = snap_mask(fine)
    W, D = legal.shape[2], legal.shape[3]
    assert (c[:, :, :W, :D][~legal] == 0).all()
    assert c[:, :, W:, :].sum() == 0 and c[:, :, :, D:].sum() == 0
    assert c.max() > 0


def test_block_cost_is_scaled_metres():
    env, plan = stage3_env(55000, 1)
    n, rot, f, x, y = plan[1]
    fine = env.pos_mask(n)
    c = block_cost(env, n, fine)
    metres = block_metres(env, env.block(n, rot))
    cx, cy = x // 2, y // 2
    fx, fy = fine_cell(fine, rot, f, cx, cy)
    assert np.isclose(c[rot, f, cx, cy], min(metres[fx, fy] / COST_SCALE, COST_CAP))


def test_block_cost_floor_maps_share_metres():
    env, plan = stage3_env(55007, 1)
    n = plan[1][0]
    fine = env.pos_mask(n)
    c = block_cost(env, n, fine)
    metres = block_metres(env, env.block(n, 0))
    for f in range(fine.shape[1]):
        assert np.allclose(c[0, f], np.minimum(snap(metres, fine[0, f]) / COST_SCALE, COST_CAP))

def test_block_cost_clips_far_squares():
    env, plan = stage3_env(55001, 0)
    n = plan[0][0]
    fine = env.pos_mask(n)
    c = block_cost(env, n, fine)
    metres = block_metres(env, env.block(n, 0))
    assert metres[fine[0].any(axis=0)].max() > COST_CAP * COST_SCALE
    assert c.max() == COST_CAP

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")