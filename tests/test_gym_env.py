import numpy as np

from layout.env import LayoutEnv
from layout.tasks import random_task, one_belt
from layout import obs as O
from layout.gym_env import (S, POS_SHAPE, N_POS, offsets, snap_mask, fine_cell, pad_mask, observe, obs_space,
    LayoutGym, play_random, a_fits, StageEnv, shaping_weight, Shaped)
from layout.testset import STAGES
from layout.baseline import greedy
from layout.api import play


PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0}
TIGHT = {"target": "Desc_IronRod_C", "rate": 90, "width": 40, "depth": 16, "floors": 3, "side": 3,
    "alternates": frozenset()}
ROOMY = {"target": "Desc_IronRod_C", "rate": 30, "width": 40, "depth": 24, "floors": 3, "side": 3,
    "alternates": frozenset()}
KEYS = {"grid", "vec", "stage", "nodes", "node_mask", "edges", "edge_rate", "edge_mask"}


def plate_env(steps):
    env = LayoutEnv()
    env.reset(dict(PLATE))
    moves = [(2, 0, 0, 2, 2), (1, 0, 1, 2, 2), (1, 0, 2, 2, 2)]
    for m in moves[:steps]:
        env.step(*m)
    return env


def act(rot, f, x, y):
    return int(np.ravel_multi_index((rot, f, x // S, y // S), POS_SHAPE))


def slow_snap(m, s):
    R, F, W, D = m.shape
    out = np.zeros((R, F, -(-W // s), -(-D // s)), dtype=bool)
    for r in range(R):
        for f in range(F):
            for cx in range(out.shape[2]):
                for cy in range(out.shape[3]):
                    out[r, f, cx, cy] = m[r, f, cx * s:cx * s + s, cy * s:cy * s + s].any()
    return out


def test_constants():
    assert S == 2
    assert POS_SHAPE == (4, 3, 48, 48)
    assert N_POS == 27648
    assert N_POS == int(np.prod(POS_SHAPE))


def test_offsets():
    assert offsets(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert offsets(1) == [(0, 0)]
    o = offsets(3)
    assert len(o) == 9 and len(set(o)) == 9
    assert o[0] == (0, 0) and o[-1] == (2, 2) and o[1:3] == [(0, 1), (1, 0)]
    assert offsets() == offsets(2)


def test_snap_mask_small():
    m = np.zeros((4, 1, 5, 4), dtype=bool)
    m[0, 0, 1, 1] = True
    m[0, 0, 2, 3] = True
    m[2, 0, 4, 0] = True
    keep = m.copy()
    c = snap_mask(m)
    assert c.shape == (4, 1, 3, 2) and c.dtype == bool
    assert c.sum() == 3
    assert c[0, 0, 0, 0] and c[0, 0, 1, 1] and c[2, 0, 2, 0]
    assert (m == keep).all()
    assert snap_mask(np.ones((4, 3, 24, 24), dtype=bool)).shape == (4, 3, 12, 12)
    assert snap_mask(np.ones((4, 3, 96, 96), dtype=bool)).shape == (4, 3, 48, 48)
    assert snap_mask(m, 1).shape == m.shape and (snap_mask(m, 1) == m).all()
    assert snap_mask(m, 4).shape == (4, 1, 2, 1) and snap_mask(m, 4).sum() == 2


def test_snap_mask_random():
    rng = np.random.default_rng(3)
    for k in range(30):
        F, W, D = int(rng.integers(1, 4)), int(rng.integers(1, 20)), int(rng.integers(1, 20))
        m = rng.random((4, F, W, D)) < rng.uniform(0, 0.3)
        for s in (1, 2, 3, 4):
            assert (snap_mask(m, s) == slow_snap(m, s)).all()


def test_fine_cell_order():
    m = np.zeros((4, 2, 5, 5), dtype=bool)
    m[1, 1, 2, 3] = True
    m[1, 1, 3, 2] = True
    assert fine_cell(m, 1, 1, 1, 1) == (2, 3)
    m[1, 1, 2, 2] = True
    assert fine_cell(m, 1, 1, 1, 1) == (2, 2)
    m[1, 1, 2, 2] = m[1, 1, 2, 3] = False
    assert fine_cell(m, 1, 1, 1, 1) == (3, 2)
    m[1, 1, 3, 3] = True
    m[1, 1, 3, 2] = False
    assert fine_cell(m, 1, 1, 1, 1) == (3, 3)
    x, y = fine_cell(m, 1, 1, 1, 1)
    assert type(x) is int and type(y) is int


def test_fine_cell_edges():
    m = np.zeros((4, 1, 5, 5), dtype=bool)
    m[0, 0, 4, 4] = True
    assert fine_cell(m, 0, 0, 2, 2) == (4, 4)
    m[0, 0, 0, 0] = True
    assert fine_cell(m, 0, 0, 0, 0) == (0, 0)
    for args in [(0, 0, 1, 1), (0, 0, 2, 1), (1, 0, 2, 2), (0, 0, 0, 2)]:
        try:
            fine_cell(m, *args)
            assert False, args
        except ValueError:
            pass
    assert fine_cell(m, 0, 0, 1, 1, s=4) == (4, 4)
    assert fine_cell(m, 0, 0, 4, 4, s=1) == (4, 4)


def test_fine_cell_plate():
    m = plate_env(0).pos_mask(2)
    assert not m[0, 0, 0, 0] and not m[0, 0, 0, 1] and m[0, 0, 1, 0]
    assert fine_cell(m, 0, 0, 0, 0) == (1, 0)
    assert fine_cell(m, 0, 0, 1, 1) == (2, 2)
    assert fine_cell(m, 0, 0, 2, 2) == (4, 4)


def test_fine_cell_random():
    rng = np.random.default_rng(4)
    for k in range(40):
        F, W, D = int(rng.integers(1, 4)), int(rng.integers(1, 12)), int(rng.integers(1, 12))
        m = rng.random((4, F, W, D)) < 0.3
        c = snap_mask(m)
        for r, f, cx, cy in np.ndindex(*c.shape):
            if c[r, f, cx, cy]:
                x, y = fine_cell(m, r, f, cx, cy)
                assert m[r, f, x, y] and x // 2 == cx and y // 2 == cy
                first = [(cx * 2 + dx, cy * 2 + dy) for dx, dy in offsets(2)
                    if cx * 2 + dx < W and cy * 2 + dy < D and m[r, f, cx * 2 + dx, cy * 2 + dy]][0]
                assert (x, y) == first
            else:
                try:
                    fine_cell(m, r, f, cx, cy)
                    assert False
                except ValueError:
                    pass


def test_pad_mask_small():
    m = np.zeros((4, 2, 24, 16), dtype=bool)
    m[0, 0, 0, 0] = True
    m[3, 1, 23, 15] = True
    m[1, 0, 5, 7] = True
    keep = m.copy()
    out = pad_mask(m)
    assert out.shape == (4, 3, 48, 48)
    assert out.dtype == bool
    assert out.sum() == 3
    assert out[0, 0, 0, 0] and out[3, 1, 23, 15] and out[1, 0, 5, 7]
    assert not out[1, 0, 7, 5]
    assert (m == keep).all()
    assert pad_mask(np.zeros((4, 3, 48, 48), dtype=bool)).sum() == 0
    full = pad_mask(np.ones((4, 3, 8, 40), dtype=bool))
    assert full.sum() == 4 * 3 * 8 * 40
    assert full[:, :, :8, :40].all()


def test_pad_mask_flat_index():
    env = plate_env(1)
    m = snap_mask(env.pos_mask(1))
    out = pad_mask(m)
    assert out.sum() == m.sum()
    flat = out.reshape(-1)
    assert flat.shape == (27648,)
    assert np.ravel_multi_index((0, 1, 1, 1), POS_SHAPE) == 2353
    assert act(0, 1, 2, 2) == 2353
    assert flat[2353]
    assert tuple(int(v) for v in np.unravel_index(2353, POS_SHAPE)) == (0, 1, 1, 1)


def test_observe_shapes():
    env = plate_env(2)
    ob = observe(env, 1, 2)
    assert set(ob) == KEYS
    shapes = {"grid": (10, 3, 96, 96), "vec": (23,), "stage": (2,), "nodes": (40, 15),
        "node_mask": (40,), "edges": (64, 2), "edge_rate": (64,), "edge_mask": (64,)}
    for k, s in shapes.items():
        assert ob[k].shape == s, k
    assert ob["grid"].dtype == np.uint8
    assert ob["stage"].dtype == np.float32
    assert ob["vec"].dtype == np.float32


def test_observe_values():
    env = plate_env(2)
    ob = observe(env, 1, 2)
    assert abs(float(ob["stage"][0]) - 1) < 1e-6
    assert abs(float(ob["stage"][1]) - 2 / 16) < 1e-6
    ob0 = observe(env, 0, 0)
    assert list(ob0["stage"]) == [0, 0]
    assert (ob["grid"] == O.grid_obs(env)).all()
    assert (ob["vec"] == O.vec_obs(env)).all()
    g = O.graph_obs(env)
    for k in g:
        assert (ob[k] == g[k]).all(), k
    assert ob["grid"].sum() == 1728 + 380 + 6 + 4


def test_obs_space_shape():
    sp = obs_space()
    assert set(sp.keys()) == KEYS
    assert sp["grid"].shape == (10, 3, 96, 96)
    assert sp["grid"].dtype == np.uint8
    assert sp["edges"].dtype == np.int32
    assert sp["node_mask"].shape == (40,)
    assert sp.contains(observe(plate_env(0), 0, 0))
    assert sp.contains(observe(plate_env(3), 0, 0))
    env = plate_env(0)
    env.world.owner[:] = 0
    ob = observe(env, 1, 16)
    assert abs(float(ob["vec"][O.V_TIGHT]) - 2) < 1e-6
    assert sp.contains(ob)


def test_obs_space_rejects():
    sp = obs_space()
    ob = observe(plate_env(2), 1, 1)
    ob["grid"] = ob["grid"].copy()
    ob["grid"][0, 0, 0, 0] = 2
    assert not sp.contains(ob)
    ob = observe(plate_env(2), 1, 1)
    ob["vec"] = ob["vec"].astype(np.float64)
    assert not sp.contains(ob)
    ob = observe(plate_env(2), 1, 1)
    ob["edges"] = ob["edges"].copy()
    ob["edges"][0, 0] = 40
    assert not sp.contains(ob)
    ob = observe(plate_env(2), 1, 1)
    del ob["stage"]
    assert not sp.contains(ob)


def test_obs_space_random():
    sp = obs_space()
    checked = 0
    for seed in range(6):
        rng = np.random.default_rng(5000 + seed)
        env = LayoutEnv()
        env.reset(random_task(rng))
        while True:
            assert sp.contains(observe(env, 0, 0))
            if env.done:
                break
            ns = np.flatnonzero(env.n_mask())
            n = int(ns[rng.integers(len(ns))])
            assert sp.contains(observe(env, 1, n))
            fine = env.pos_mask(n)
            m = pad_mask(snap_mask(fine))
            assert m.sum() == snap_mask(fine).sum() <= fine.sum()
            spots = np.argwhere(m)
            rot, f, cx, cy = (int(v) for v in spots[rng.integers(len(spots))])
            x, y = fine_cell(fine, rot, f, cx, cy)
            assert m.reshape(-1)[act(rot, f, x, y)]
            env.step(n, rot, f, x, y)
            checked += 1
    assert checked >= 20


def plate_gym():
    g = LayoutGym()
    obs, info = g.reset(options={"task": dict(PLATE)})
    return g, obs, info


def test_reset_plate():
    g, obs, info = plate_gym()
    assert set(obs) == KEYS
    assert info == {"stage": 0, "failed": False, "blocks": 0}
    m = g.action_masks()
    assert m.shape == (27648,) and m.dtype == bool
    assert list(np.flatnonzero(m)) == [1, 2]
    assert list(obs["stage"]) == [0, 0]


def test_odd_corner_step():
    task = dict(PLATE)
    task["width"] = 25
    g = LayoutGym()
    g.reset(options={"task": task})
    g.step(2)
    assert g.action_masks().reshape(POS_SHAPE)[0, 0, :13].any()
    assert not g.action_masks().reshape(POS_SHAPE)[0, 0, 13:].any()
    g.step(act(0, 0, 0, 0))
    b = g.env.world.blocks[0]
    assert (b["floor"], b["x"], b["y"]) == (0, 1, 0)


def test_two_stage_step():
    g, obs, info = plate_gym()
    obs, r, term, trunc, info = g.step(2)
    assert (r, term, trunc) == (0.0, False, False)
    assert info["stage"] == 1 and info["blocks"] == 0
    assert abs(float(obs["stage"][1]) - 2 / 16) < 1e-6
    m = g.action_masks()
    assert m.sum() == snap_mask(g.env.pos_mask(2)).sum() == 244
    assert (g.fine == g.env.pos_mask(2)).all()
    assert m[act(0, 0, 2, 2)]
    obs, r, term, trunc, info = g.step(act(0, 0, 2, 2))
    assert (r, term) == (0.0, False)
    assert info == {"stage": 0, "failed": False, "blocks": 1, "wire": 9 / 96}
    assert list(obs["stage"]) == [0, 0]
    assert g.env.world.blocks[0]["floor"] == 0 and g.env.world.blocks[0]["x"] == 2
    assert list(np.flatnonzero(g.action_masks())) == [1, 2]


def test_full_plate_episode():
    g, obs, info = plate_gym()
    for a in [2, act(0, 0, 2, 2), 1, act(0, 1, 2, 2), 1, act(0, 2, 2, 2)]:
        obs, r, term, trunc, info = g.step(a)
    plan = [[b["block"]["n"], b["block"]["rot"], b["floor"], b["x"], b["y"]] for b in g.env.world.blocks]
    assert plan == [[2, 0, 0, 2, 2], [1, 0, 1, 2, 2], [1, 0, 2, 2, 2]]
    assert term and abs(r - 12 / 19) < 1e-9 and not info["failed"]
    assert info["terms"]["length"] == 64 and abs(info["terms"]["penalty"] - 7 / 12) < 1e-9
    assert info["blocks"] == 3 and info["wire_total"] == 9 + 77 + 109
    assert g.action_masks().sum() == 0
    assert obs["grid"][1].sum() == 200 + 180 + 180
    try:
        g.step(1)
        assert False
    except ValueError:
        pass


def test_illegal_actions():
    g, obs, info = plate_gym()
    for a in [0, 3, 16, -1, 27648, 2353]:
        try:
            g.step(a)
            assert False, a
        except ValueError:
            pass
    assert g.stage == 0 and g.action_masks().sum() == 2
    g.step(1)
    for a in [act(0, 0, 30, 2), act(0, 0, 46, 46), act(3, 2, 94, 94), 27648]:
        try:
            g.step(a)
            assert False, a
        except ValueError:
            pass
    assert g.stage == 1 and g.n == 1 and len(g.env.world.blocks) == 0


def test_reset_seed():
    a, b = LayoutGym(), LayoutGym()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert a.task == b.task
    assert all((oa[k] == ob[k]).all() for k in KEYS)
    c = LayoutGym()
    c.reset(seed=8)
    d = LayoutGym()
    d.reset(seed=9)
    assert c.task != d.task
    a.step(int(np.flatnonzero(a.action_masks())[0]))
    a.reset(seed=7)
    assert a.task == b.task and a.stage == 0 and len(a.env.world.blocks) == 0


def test_random_play_invariants():
    g = LayoutGym()
    rng = np.random.default_rng(1)
    for ep in range(5):
        obs, info = g.reset(seed=600 + ep)
        steps = 0
        term = False
        while not term:
            assert g.observation_space.contains(obs)
            m = g.action_masks()
            assert m.any()
            if g.stage == 0:
                assert not m[0] and not m[17:].any()
            legal = np.flatnonzero(m)
            obs, r, term, trunc, info = g.step(int(legal[rng.integers(len(legal))]))
            steps += 1
            assert term == g.env.done and not trunc
            if not term:
                assert r == 0.0
        assert g.observation_space.contains(obs)
        assert steps == 2 * info["blocks"]
        assert "terms" in info
        if info["failed"]:
            assert -2 <= r < -1
        elif info["terms"]["passed"]:
            assert 0 < r <= 1
        else:
            assert -1 <= r < 0


def test_play_random():
    res = play_random(4, seed=3)
    assert res["episodes"] == 4
    assert res["finished"] + res["failed"] == 4
    assert res["steps"] >= 8
    assert res["valid"] <= res["finished"]
    assert -2 <= res["mean_score"] <= 1


def test_a_fits_hand():
    assert a_fits(dict(TIGHT)) is False
    assert a_fits(dict(ROOMY)) is True


def test_a_fits_needs_both_rotations():
    fixed_only = StageEnv(1).task(np.random.default_rng(4))
    free_only = StageEnv(2).task(np.random.default_rng(58))
    assert play(fixed_only, greedy(fixed_only, True)).failed
    assert play(free_only, greedy(free_only, False)).failed
    assert a_fits(fixed_only) and a_fits(free_only)


def test_stage_env_default_unchanged():
    env = StageEnv(2)
    assert env.solvable is False
    for seed in range(8):
        rng = np.random.default_rng(seed)
        target = STAGES[1][rng.integers(len(STAGES[1]))]
        assert env.task(np.random.default_rng(seed)) == random_task(rng, target)


def test_stage_env_solvable_redraws():
    plain, solv = StageEnv(2), StageEnv(2, solvable=True)
    assert not a_fits(plain.task(np.random.default_rng(3)))
    t = solv.task(np.random.default_rng(3))
    assert a_fits(t) and t["target"] in STAGES[1]
    assert solv.task(np.random.default_rng(3)) == t
    assert solv.task(np.random.default_rng(0)) == plain.task(np.random.default_rng(0))


def test_stage_env_solvable_all_fit():
    env = StageEnv(2, solvable=True)
    for seed in range(12):
        assert a_fits(env.task(np.random.default_rng(seed)))


def test_stage_env_skips_over_belt_tasks():
    rng = np.random.default_rng(1_004_236)
    bad = random_task(rng, STAGES[2][rng.integers(len(STAGES[2]))])
    assert one_belt(bad) is False
    for env in (StageEnv(3), StageEnv(3, solvable=True)):
        t = env.task(np.random.default_rng(1_004_236))
        assert t != bad and one_belt(t) and t["target"] in STAGES[2]
        assert env.task(np.random.default_rng(1_004_236)) == t
    assert a_fits(StageEnv(3, solvable=True).task(np.random.default_rng(1_004_236)))


def test_wire_info_plate():
    g, obs, info = plate_gym()
    wires, rewards = [], []
    for a in [2, act(0, 0, 2, 2), 1, act(0, 1, 2, 2), 1, act(0, 2, 2, 2)]:
        obs, r, term, trunc, info = g.step(a)
        wires.append(info["wire"])
        rewards.append(r)
    assert np.allclose(wires, [0, 9 / 96, 0, 77 / 96, 0, 109 / 96])
    assert rewards[:5] == [0.0] * 5 and abs(rewards[5] - 12 / 19) < 1e-9
    assert info["wire_total"] == 195


def test_wire_total_restarts_on_reset():
    g, obs, info = plate_gym()
    for a in [2, act(0, 0, 2, 2), 1, act(0, 1, 2, 2), 1, act(0, 2, 2, 2)]:
        g.step(a)
    g.reset(options={"task": dict(PLATE)})
    for a in [2, act(0, 0, 2, 2), 1, act(0, 1, 2, 2), 1, act(0, 2, 2, 2)]:
        obs, r, term, trunc, info = g.step(a)
    assert info["wire_total"] == 195


def test_shaping_weight_schedule():
    assert shaping_weight(0, 2000, 0.1) == 0.1
    assert abs(shaping_weight(500, 2000, 0.1) - 0.05) < 1e-12
    assert shaping_weight(1000, 2000, 0.1) == 0.0
    assert shaping_weight(1500, 2000, 0.1) == 0.0
    assert abs(shaping_weight(250, 1000, 0.2) - 0.1) < 1e-12
    assert shaping_weight(7, 2000, 0.0) == 0.0
    assert isinstance(shaping_weight(1200, 2000, 0.1), float)


class FakeGames:
    def __init__(self):
        self.seen = None

    def reset(self, seed=0):
        self.seen = seed
        return ["o"], np.ones((1, 3), dtype=bool)

    def step(self, actions):
        infos = [{"wire": 0.5}, {"stage": 1}, {"wire": 2.0, "score": 0.7}]
        return ["a", "b", "c"], np.ones((3, 3), dtype=bool), np.array([0.0, 0.0, 0.7], dtype=np.float32), \
            np.array([False, False, True]), infos


def test_shaped_subtracts_weight_times_wire():
    games = Shaped(FakeGames(), weight=0.1)
    obs, masks, rewards, dones, infos = games.step([0, 0, 0])
    assert np.allclose(rewards, [-0.05, 0.0, 0.5]) and rewards.dtype == np.float32
    assert infos[2]["score"] == 0.7 and obs == ["a", "b", "c"] and list(dones) == [False, False, True]


def test_shaped_weight_zero_and_changes():
    games = Shaped(FakeGames())
    assert np.allclose(games.step([0])[2], [0.0, 0.0, 0.7])
    games.weight = 0.2
    assert np.allclose(games.step([0])[2], [-0.1, 0.0, 0.3])
    assert games.reset(5)[0] == ["o"] and games.games.seen == 5


TESTS = [test_constants, test_offsets, test_snap_mask_small, test_snap_mask_random, test_fine_cell_order,
    test_fine_cell_edges, test_fine_cell_plate, test_fine_cell_random, test_pad_mask_small, test_pad_mask_flat_index,
    test_observe_shapes, test_observe_values, test_obs_space_shape, test_obs_space_rejects, test_obs_space_random,
    test_reset_plate, test_odd_corner_step, test_two_stage_step, test_full_plate_episode, test_illegal_actions, test_reset_seed,
    test_random_play_invariants, test_play_random, test_a_fits_hand, test_a_fits_needs_both_rotations,
    test_stage_env_default_unchanged,
    test_stage_env_solvable_redraws, test_stage_env_solvable_all_fit, test_stage_env_skips_over_belt_tasks,
    test_wire_info_plate,
    test_wire_total_restarts_on_reset, test_shaping_weight_schedule, test_shaped_subtracts_weight_times_wire,
    test_shaped_weight_zero_and_changes]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")