import numpy as np

from layout.env import LayoutEnv
from layout.tasks import random_task
from layout import obs as O
from layout.gym_env import POS_SHAPE, N_POS, pad_mask, observe, obs_space, LayoutGym, play_random


PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0}
KEYS = {"grid", "vec", "stage", "nodes", "node_mask", "edges", "edge_rate", "edge_mask"}


def plate_env(steps):
    env = LayoutEnv()
    env.reset(dict(PLATE))
    moves = [(2, 0, 0, 2, 2), (1, 0, 1, 2, 2), (1, 0, 2, 2, 2)]
    for m in moves[:steps]:
        env.step(*m)
    return env


def test_constants():
    assert POS_SHAPE == (4, 3, 96, 96)
    assert N_POS == 110592
    assert N_POS == int(np.prod(POS_SHAPE))


def test_pad_mask_small():
    m = np.zeros((4, 2, 24, 16), dtype=bool)
    m[0, 0, 0, 0] = True
    m[3, 1, 23, 15] = True
    m[1, 0, 5, 7] = True
    keep = m.copy()
    out = pad_mask(m)
    assert out.shape == (4, 3, 96, 96)
    assert out.dtype == bool
    assert out.sum() == 3
    assert out[0, 0, 0, 0] and out[3, 1, 23, 15] and out[1, 0, 5, 7]
    assert not out[1, 0, 7, 5]
    assert (m == keep).all()
    assert pad_mask(np.zeros((4, 3, 96, 96), dtype=bool)).sum() == 0
    full = pad_mask(np.ones((4, 3, 8, 40), dtype=bool))
    assert full.sum() == 4 * 3 * 8 * 40
    assert full[:, :, :8, :40].all()


def test_pad_mask_flat_index():
    env = plate_env(1)
    m = env.pos_mask(1)
    out = pad_mask(m)
    assert out.sum() == m.sum()
    flat = out.reshape(-1)
    assert flat.shape == (110592,)
    assert flat[9410]
    assert np.ravel_multi_index((0, 1, 2, 2), POS_SHAPE) == 9410
    assert tuple(int(v) for v in np.unravel_index(9410, POS_SHAPE)) == (0, 1, 2, 2)


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
            m = pad_mask(env.pos_mask(n))
            assert m.sum() == env.pos_mask(n).sum()
            spots = np.argwhere(m)
            rot, f, x, y = (int(v) for v in spots[rng.integers(len(spots))])
            a = int(np.ravel_multi_index((rot, f, x, y), POS_SHAPE))
            assert m.reshape(-1)[a]
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
    assert m.shape == (110592,) and m.dtype == bool
    assert list(np.flatnonzero(m)) == [1, 2]
    assert list(obs["stage"]) == [0, 0]


def test_two_stage_step():
    g, obs, info = plate_gym()
    obs, r, term, trunc, info = g.step(2)
    assert (r, term, trunc) == (0.0, False, False)
    assert info["stage"] == 1 and info["blocks"] == 0
    assert abs(float(obs["stage"][1]) - 2 / 16) < 1e-6
    m = g.action_masks()
    assert m.sum() == g.env.pos_mask(2).sum()
    assert m[np.ravel_multi_index((0, 0, 2, 2), POS_SHAPE)]
    obs, r, term, trunc, info = g.step(np.ravel_multi_index((0, 0, 2, 2), POS_SHAPE))
    assert (r, term) == (0.0, False)
    assert info == {"stage": 0, "failed": False, "blocks": 1}
    assert list(obs["stage"]) == [0, 0]
    assert g.env.world.blocks[0]["floor"] == 0 and g.env.world.blocks[0]["x"] == 2
    assert list(np.flatnonzero(g.action_masks())) == [1, 2]


def test_full_plate_episode():
    g, obs, info = plate_gym()
    for a in [2, 9410 - 9216, 1, 9410, 1, 9410 + 9216]:
        obs, r, term, trunc, info = g.step(a)
    assert term and abs(r - 12 / 19) < 1e-9 and not info["failed"]
    assert info["terms"]["length"] == 64 and abs(info["terms"]["penalty"] - 7 / 12) < 1e-9
    assert info["blocks"] == 3
    assert g.action_masks().sum() == 0
    assert obs["grid"][1].sum() == 200 + 180 + 180
    try:
        g.step(1)
        assert False
    except ValueError:
        pass


def test_illegal_actions():
    g, obs, info = plate_gym()
    for a in [0, 3, 16, -1, 110592, 9410]:
        try:
            g.step(a)
            assert False, a
        except ValueError:
            pass
    assert g.stage == 0 and g.action_masks().sum() == 2
    g.step(1)
    for a in [1, 2, np.ravel_multi_index((0, 0, 30, 2), POS_SHAPE)]:
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


TESTS = [test_constants, test_pad_mask_small, test_pad_mask_flat_index, test_observe_shapes,
    test_observe_values, test_obs_space_shape, test_obs_space_rejects, test_obs_space_random,
    test_reset_plate, test_two_stage_step, test_full_plate_episode, test_illegal_actions, test_reset_seed,
    test_random_play_invariants, test_play_random]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")