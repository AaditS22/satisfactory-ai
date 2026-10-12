import numpy as np

from solver.gamedata import recipe_by_id
from solver.lp_solve import allowed_recipes
from layout.env import LayoutEnv
from layout.tasks import random_task
from layout.obs import (SITE, BLOCK, PORT, NEED, MAX_INPUTS, GIVE, SAME, OPEN, N_CH,
    paint, base_layers, head_layers, grid_obs)
from layout import obs as O


PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0}


def plate_env(steps):
    env = LayoutEnv()
    env.reset(dict(PLATE))
    moves = [(2, 0, 0, 2, 2), (1, 0, 1, 2, 2), (1, 0, 2, 2, 2)]
    for m in moves[:steps]:
        env.step(*m)
    return env


def blank():
    return np.zeros((N_CH, 3, 96, 96), dtype=np.uint8)


def cells(layer):
    return {tuple(int(v) for v in c) for c in np.argwhere(layer)}


def test_constants():
    chans = [SITE, BLOCK, PORT] + [NEED + k for k in range(MAX_INPUTS)] + [GIVE, SAME, OPEN]
    assert sorted(chans) == list(range(N_CH))
    assert N_CH == 10
    assert max(len(r["inputs"]) for r in allowed_recipes(None)) <= MAX_INPUTS


def test_paint():
    obs = blank()
    paint(obs, 4, [(0, 1, 2), (2, 95, 0), (0, 1, 2)])
    assert obs.sum() == 2
    assert obs[4, 0, 1, 2] == 1
    assert obs[4, 2, 95, 0] == 1
    assert obs[4, 0, 2, 1] == 0
    paint(obs, 5, [])
    assert obs.sum() == 2
    assert obs.dtype == np.uint8


def test_base_fresh():
    env = plate_env(0)
    obs = blank()
    base_layers(env, obs)
    assert obs[SITE].sum() == 3 * 24 * 24
    assert obs[SITE, 2, 23, 23] == 1
    assert obs[SITE, 0, 24, 0] == 0
    assert obs[SITE, 0, 0, 24] == 0
    assert obs[BLOCK].sum() == 0
    assert cells(obs[PORT]) == {(0, 0, 12), (0, 23, 12)}
    assert obs[3:].sum() == 0


def test_base_trace():
    env = plate_env(2)
    obs = blank()
    base_layers(env, obs)
    assert obs[BLOCK].sum() == 10 * 20 + 9 * 20
    assert obs[BLOCK, 0, 2, 2] == 1
    assert obs[BLOCK, 0, 11, 21] == 1
    assert obs[BLOCK, 0, 12, 2] == 0
    assert obs[BLOCK, 1, 10, 2] == 1
    assert obs[BLOCK, 1, 11, 2] == 0
    assert obs[BLOCK, 2].sum() == 0
    assert cells(obs[PORT]) == {(0, 0, 12), (0, 23, 12), (0, 1, 4), (0, 12, 19), (1, 1, 4), (1, 11, 19)}


def test_head_first_step():
    env = plate_env(0)
    obs = blank()
    head_layers(env, obs)
    assert cells(obs[NEED]) == {(0, 0, 12)}
    assert obs[NEED + 1:NEED + MAX_INPUTS].sum() == 0
    assert obs[GIVE].sum() == 0
    assert obs[SAME].sum() == 0
    assert cells(obs[OPEN]) == {(0, 0, 12)}
    assert obs[:3].sum() == 0


def test_head_trace():
    env = plate_env(2)
    obs = blank()
    head_layers(env, obs)
    assert cells(obs[NEED]) == {(0, 12, 19)}
    assert obs[NEED + 1:NEED + MAX_INPUTS].sum() == 0
    assert cells(obs[GIVE]) == {(0, 23, 12)}
    assert cells(obs[SAME]) == {(1, 11, 19)}
    assert cells(obs[OPEN]) == {(0, 12, 19)}


def test_grid_obs_trace():
    env = plate_env(2)
    owner = env.world.owner.copy()
    reserved = env.world.reserved.copy()
    q = [dict(e) for e in env.queue]
    obs = grid_obs(env)
    assert obs.shape == (10, 3, 96, 96)
    assert obs.dtype == np.uint8
    assert obs.sum() == 1728 + 380 + 6 + 4
    assert set(np.unique(obs)) <= {0, 1}
    assert (env.world.owner == owner).all()
    assert (env.world.reserved == reserved).all()
    assert env.queue == q
    again = grid_obs(env)
    again[SITE] = 0
    assert obs[SITE].sum() == 1728


def test_grid_obs_done():
    env = plate_env(3)
    assert env.done and not env.failed
    obs = grid_obs(env)
    assert obs[3:].sum() == 0
    assert obs[BLOCK].sum() == 10 * 20 + 9 * 20 * 2
    assert obs[PORT].sum() == 8


def slow_obs(env):
    obs = np.zeros((N_CH, 3, 96, 96), dtype=np.uint8)
    F, W, D = env.world.owner.shape
    for f in range(F):
        for x in range(W):
            for y in range(D):
                obs[SITE, f, x, y] = 1
                if env.world.owner[f, x, y] >= 0:
                    obs[BLOCK, f, x, y] = 1
                if env.world.reserved[f, x, y]:
                    obs[PORT, f, x, y] = 1
    if not env.queue:
        return obs
    rec = recipe_by_id[env.queue[0]["recipe"]]
    ins = list(rec["inputs"])
    out = list(rec["outputs"])[0]
    wanted = set()
    for q in env.queue:
        wanted |= set(recipe_by_id[q["recipe"]]["inputs"])
    for b in env.world.blocks:
        f = b["floor"]
        bout = b["block"]["output"]["item"]
        ox, oy = b["ends"][-1]
        for k, it in enumerate(ins):
            if bout == it:
                obs[NEED + k, f, ox, oy] = 1
        if bout == out:
            obs[SAME, f, ox, oy] = 1
        if bout in wanted:
            obs[OPEN, f, ox, oy] = 1
        for k, inp in enumerate(b["block"]["inputs"]):
            if inp["item"] == out:
                ix, iy = b["ends"][k]
                obs[GIVE, f, ix, iy] = 1
    for it, (x, y) in env.entries.items():
        for k, i2 in enumerate(ins):
            if i2 == it:
                obs[NEED + k, 0, x, y] = 1
        if it in wanted:
            obs[OPEN, 0, x, y] = 1
    if out == env.graph["target"]:
        obs[GIVE, 0, env.exit[0], env.exit[1]] = 1
    return obs


def test_random_vs_slow():
    checked = 0
    for seed in range(6):
        rng = np.random.default_rng(1000 + seed)
        env = LayoutEnv()
        env.reset(random_task(rng))
        while True:
            obs = grid_obs(env)
            assert (obs == slow_obs(env)).all()
            assert (obs[BLOCK] <= obs[SITE]).all()
            assert (obs[BLOCK] & obs[PORT]).sum() == 0
            for ch in list(range(NEED, NEED + MAX_INPUTS)) + [GIVE, SAME, OPEN]:
                assert (obs[ch] <= obs[PORT]).all()
            for k in range(MAX_INPUTS):
                assert (obs[NEED + k] <= obs[OPEN]).all()
            checked += 1
            if env.done:
                break
            ns = np.flatnonzero(env.n_mask())
            n = int(ns[rng.integers(len(ns))])
            spots = np.argwhere(env.pos_mask(n))
            env.step(n, *(int(v) for v in spots[rng.integers(len(spots))]))
    assert checked >= 20


def vblank():
    return np.zeros(O.N_VEC, dtype=np.float32)


def close(a, b):
    return abs(float(a) - float(b)) < 1e-6


def test_vec_constants():
    assert O.N_VEC == 23
    assert O.MACHINES == sorted(O.MACHINES)
    assert len(O.MACHINES) == 5
    assert {r["machine"] for r in allowed_recipes(None)} <= set(O.MACHINES)
    idx = list(range(O.V_MACH, O.V_MACH + 5)) + list(range(O.V_SLOT, O.V_SLOT + MAX_INPUTS))
    idx += [O.V_W, O.V_D, O.V_NLIM, O.V_REM, O.V_CLOCK, O.V_TARGET, O.V_QLEN, O.V_MLEFT]
    idx += list(range(O.V_FREE, O.V_FREE + 3)) + [O.V_TIGHT, O.V_WIDTH, O.V_DEPTH]
    assert sorted(idx) == list(range(O.N_VEC))


def test_free_cells():
    assert list(O.free_cells(plate_env(0))) == [574, 576, 576]
    assert list(O.free_cells(plate_env(2))) == [372, 394, 576]
    assert list(O.free_cells(plate_env(3))) == [372, 394, 394]


def test_head_vec_reset():
    vec = vblank()
    O.head_vec(plate_env(0), vec)
    assert vec[O.V_MACH + 4] == 1 and vec[O.V_MACH:O.V_MACH + 5].sum() == 1
    assert list(vec[O.V_SLOT:O.V_SLOT + 4]) == [1, 0, 0, 0]
    assert close(vec[O.V_W], 5 / 96)
    assert close(vec[O.V_D], 20 / 96)
    assert close(vec[O.V_NLIM], 2 / 16)
    assert close(vec[O.V_REM], 2 / 48)
    assert close(vec[O.V_CLOCK], 0.75)
    assert vec[O.V_TARGET] == 0
    assert vec[O.V_QLEN:].sum() == 0
    env = LayoutEnv()
    env.reset({"target": "Desc_IronPlate_C", "rate": 400, "width": 96, "depth": 96, "floors": 3, "side": 0})
    vec = vblank()
    O.head_vec(env, vec)
    assert close(vec[O.V_NLIM], 1.0)
    assert close(vec[O.V_REM], 20 / 48)
    assert close(vec[O.V_CLOCK], 1.0)


def test_head_vec_trace():
    vec = vblank()
    O.head_vec(plate_env(2), vec)
    assert vec[O.V_MACH + 1] == 1 and vec[O.V_MACH:O.V_MACH + 5].sum() == 1
    assert list(vec[O.V_SLOT:O.V_SLOT + 4]) == [1, 0, 0, 0]
    assert close(vec[O.V_W], 9 / 96)
    assert close(vec[O.V_D], 20 / 96)
    assert close(vec[O.V_NLIM], 1 / 16)
    assert close(vec[O.V_REM], 1 / 48)
    assert vec[O.V_TARGET] == 1


def test_head_vec_done():
    vec = vblank()
    O.head_vec(plate_env(3), vec)
    assert vec.sum() == 0


def test_queue_vec():
    vec = vblank()
    O.queue_vec(plate_env(0), vec)
    assert close(vec[O.V_QLEN], 2 / 32)
    assert close(vec[O.V_MLEFT], 4 / 48)
    assert close(vec[O.V_FREE], 574 / 576)
    assert close(vec[O.V_FREE + 1], 1.0)
    assert close(vec[O.V_TIGHT], 560 / 1726)
    assert close(vec[O.V_WIDTH], 0.25) and close(vec[O.V_DEPTH], 0.25)
    assert vec[:O.V_QLEN].sum() == 0
    vec = vblank()
    O.queue_vec(plate_env(2), vec)
    assert close(vec[O.V_QLEN], 1 / 32)
    assert close(vec[O.V_MLEFT], 1 / 48)
    assert close(vec[O.V_FREE], 372 / 576)
    assert close(vec[O.V_FREE + 1], 394 / 576)
    assert close(vec[O.V_FREE + 2], 1.0)
    assert close(vec[O.V_TIGHT], 180 / 1342)
    vec = vblank()
    O.queue_vec(plate_env(3), vec)
    assert vec[O.V_QLEN] == 0 and vec[O.V_MLEFT] == 0 and vec[O.V_TIGHT] == 0
    assert close(vec[O.V_FREE + 2], 394 / 576)


def test_vec_obs_trace():
    env = plate_env(2)
    q = [dict(e) for e in env.queue]
    vec = O.vec_obs(env)
    assert vec.shape == (23,)
    assert vec.dtype == np.float32
    want = [0, 1, 0, 0, 0, 1, 0, 0, 0, 9 / 96, 20 / 96, 1 / 16, 1 / 48, 0.75, 1,
        1 / 32, 1 / 48, 372 / 576, 394 / 576, 1, 180 / 1342, 0.25, 0.25]
    assert all(close(a, b) for a, b in zip(vec, want))
    assert env.queue == q
    assert O.vec_obs(env) is not vec


def test_vec_random():
    from layout.blocks import make_block
    checked = 0
    for seed in range(6):
        rng = np.random.default_rng(2000 + seed)
        env = LayoutEnv()
        env.reset(random_task(rng))
        while True:
            vec = O.vec_obs(env)
            F, W, D = env.world.owner.shape
            assert vec.min() >= 0 and vec.max() <= 2
            for f in range(F):
                free = 0
                for x in range(W):
                    for y in range(D):
                        if env.world.owner[f, x, y] < 0 and not env.world.reserved[f, x, y]:
                            free += 1
                assert close(vec[O.V_FREE + f], free / (W * D))
            if env.queue:
                head = env.queue[0]
                rec = recipe_by_id[head["recipe"]]
                assert vec[O.V_MACH:O.V_MACH + 5].sum() == 1
                assert vec[O.V_MACH + O.MACHINES.index(rec["machine"])] == 1
                assert vec[O.V_SLOT:O.V_SLOT + 4].sum() == len(rec["inputs"])
                b = make_block(head["recipe"], 3, head["clock"])
                assert close(vec[O.V_W] * 96 * 3, b["w"])
                assert close(vec[O.V_NLIM] * 16, env.n_limit())
                need = sum(q["remaining"] * make_block(q["recipe"], 1, q["clock"])["w"]
                    * make_block(q["recipe"], 1, q["clock"])["d"] for q in env.queue)
                total = (vec[O.V_FREE:O.V_FREE + 3] * W * D).sum()
                assert close(vec[O.V_TIGHT], min(need / max(total, 1), 2.0))
                assert close(vec[O.V_MLEFT] * 48, sum(q["remaining"] for q in env.queue))
            else:
                assert vec[:O.V_QLEN].sum() == 0
            checked += 1
            if env.done:
                break
            ns = np.flatnonzero(env.n_mask())
            n = int(ns[rng.integers(len(ns))])
            spots = np.argwhere(env.pos_mask(n))
            env.step(n, *(int(v) for v in spots[rng.integers(len(spots))]))
    assert checked >= 20


def test_tight_cap():
    env = LayoutEnv()
    env.reset({"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0})
    env.world.owner[:] = 0
    env.world.owner[0, 5, 5] = -1
    vec = vblank()
    O.queue_vec(env, vec)
    assert close(vec[O.V_TIGHT], 2.0)
    env.world.owner[0, 5, 5] = 0
    vec = vblank()
    O.queue_vec(env, vec)
    assert close(vec[O.V_TIGHT], 2.0)
    assert vec[O.V_FREE:O.V_FREE + 3].sum() == 0


ING, PL, ORE = "Recipe_IngotIron_C", "Recipe_IronPlate_C", "Desc_OreIron_C"


def rows_close(a, b):
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def test_graph_constants():
    assert O.N_NODE == 15
    idx = list(range(O.G_KIND, O.G_KIND + 3)) + list(range(O.G_MACH, O.G_MACH + 5))
    idx += [O.G_TOTAL, O.G_PLACED, O.G_HEAD, O.G_DEPTH] + list(range(O.G_POS, O.G_POS + 3))
    assert sorted(idx) == list(range(O.N_NODE))
    assert O.MAX_NODES >= 24 + 8 + 1
    assert O.MAX_EDGES >= 47


def test_node_list():
    assert O.node_list(plate_env(0)) == [ING, PL, ORE, "OUTPUT"]
    assert O.node_list(plate_env(3)) == [ING, PL, ORE, "OUTPUT"]
    for seed in range(5):
        env = LayoutEnv()
        env.reset(random_task(np.random.default_rng(3000 + seed)))
        keys = O.node_list(env)
        R = len(env.graph["nodes"])
        assert keys[:R] == [q["recipe"] for q in env.queue]
        assert keys[R:-1] == sorted(env.entries)
        assert keys[-1] == "OUTPUT"
        assert len(keys) == len(set(keys)) <= O.MAX_NODES


def test_placed():
    assert O.placed(plate_env(0)) == {}
    assert O.placed(plate_env(2)) == {ING: {"machines": 2, "ends": [(0, 12, 19)]},
        PL: {"machines": 1, "ends": [(1, 11, 19)]}}
    assert O.placed(plate_env(3))[PL] == {"machines": 2, "ends": [(1, 11, 19), (2, 11, 19)]}


def test_node_feats_reset():
    env = plate_env(0)
    f = O.node_feats(env, O.node_list(env))
    assert f.shape == (O.MAX_NODES, 15) and f.dtype == np.float32
    assert rows_close(f[0], [1, 0, 0, 0, 0, 0, 0, 1, 2 / 48, 0, 1, 0.1, 0, 0, 0])
    assert rows_close(f[1], [1, 0, 0, 0, 1, 0, 0, 0, 2 / 48, 0, 0, 0, 0, 0, 0])
    assert rows_close(f[2], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 12 / 96])
    assert rows_close(f[3], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 23 / 96, 12 / 96])
    assert f[4:].sum() == 0


def test_node_feats_trace():
    env = plate_env(2)
    f = O.node_feats(env, O.node_list(env))
    assert rows_close(f[0], [1, 0, 0, 0, 0, 0, 0, 1, 2 / 48, 1, 0, 0.1, 0, 12 / 96, 19 / 96])
    assert rows_close(f[1], [1, 0, 0, 0, 1, 0, 0, 0, 2 / 48, 0.5, 1, 0, 0.5, 11 / 96, 19 / 96])
    env = plate_env(3)
    f = O.node_feats(env, O.node_list(env))
    assert rows_close(f[1], [1, 0, 0, 0, 1, 0, 0, 0, 2 / 48, 1, 0, 0, 0.75, 11 / 96, 19 / 96])
    assert f[:, O.G_HEAD].sum() == 0


def test_edge_arrays():
    env = plate_env(2)
    keys = O.node_list(env)
    pairs, rate = O.edge_arrays(env, {k: i for i, k in enumerate(keys)})
    assert pairs.shape == (O.MAX_EDGES, 2) and pairs.dtype == np.int32
    assert rate.shape == (O.MAX_EDGES,) and rate.dtype == np.float32
    got = {(int(a), int(b)): float(r) for (a, b), r in zip(pairs[:3], rate[:3])}
    assert set(got) == {(1, 3), (0, 1), (2, 0)}
    assert close(got[(1, 3)], 30 / 1200)
    assert close(got[(0, 1)], 45 / 1200)
    assert close(got[(2, 0)], 45 / 1200)
    assert pairs[3:].sum() == 0 and rate[3:].sum() == 0


def test_graph_obs_trace():
    env = plate_env(2)
    q = [dict(e) for e in env.queue]
    g = O.graph_obs(env)
    assert set(g) == {"nodes", "node_mask", "edges", "edge_rate", "edge_mask"}
    assert g["node_mask"].dtype == bool and g["edge_mask"].dtype == bool
    assert list(np.flatnonzero(g["node_mask"])) == [0, 1, 2, 3]
    assert list(np.flatnonzero(g["edge_mask"])) == [0, 1, 2]
    assert g["nodes"].shape == (O.MAX_NODES, 15)
    assert close(g["nodes"][1, O.G_PLACED], 0.5)
    assert env.queue == q


def test_graph_random():
    for seed in range(6):
        rng = np.random.default_rng(4000 + seed)
        env = LayoutEnv()
        env.reset(random_task(rng))
        while True:
            g = O.graph_obs(env)
            keys = O.node_list(env)
            N, E = len(keys), len(env.graph["edges"])
            nodes = g["nodes"]
            assert g["node_mask"].sum() == N and g["node_mask"][:N].all()
            assert g["edge_mask"].sum() == E and g["edge_mask"][:E].all()
            assert nodes[N:].sum() == 0 and g["edges"][E:].sum() == 0
            assert (nodes[:N, 0:3].sum(axis=1) == 1).all()
            assert nodes.min() >= 0 and nodes.max() <= 1
            left = {q["recipe"]: q["remaining"] for q in env.queue}
            for i, k in enumerate(keys[:len(env.graph["nodes"])]):
                total = len(env.graph["nodes"][k]["clocks"])
                assert close(nodes[i, O.G_PLACED], 1 - left.get(k, 0) / total)
                assert nodes[i, O.G_HEAD] == (1 if env.queue and env.queue[0]["recipe"] == k else 0)
            for j, e in enumerate(env.graph["edges"]):
                a, b = (int(v) for v in g["edges"][j])
                assert keys[b] == e["to"]
                assert keys[a] == (e["item"] if e["from"] == "RAW" else e["from"])
                assert abs(float(g["edge_rate"][j]) - e["rate"] / 1200) < 1e-6
            if env.done:
                break
            ns = np.flatnonzero(env.n_mask())
            n = int(ns[rng.integers(len(ns))])
            spots = np.argwhere(env.pos_mask(n))
            env.step(n, *(int(v) for v in spots[rng.integers(len(spots))]))


TESTS = [test_constants, test_paint, test_base_fresh, test_base_trace, test_head_first_step,
    test_head_trace, test_grid_obs_trace, test_grid_obs_done, test_random_vs_slow, test_vec_constants, test_free_cells, test_head_vec_reset,
    test_head_vec_trace, test_head_vec_done, test_queue_vec, test_vec_obs_trace, test_vec_random, test_tight_cap,
    test_graph_constants, test_node_list, test_placed, test_node_feats_reset, test_node_feats_trace,
    test_edge_arrays, test_graph_obs_trace, test_graph_random]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")