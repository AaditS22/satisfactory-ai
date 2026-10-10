import copy
import random

import numpy as np

from solver.graph import build_graph
from layout.env import N_MAX, LayoutEnv, edge_cells, raw_items
from layout.world import EMPTY, end_cells

RIP = "Desc_IronPlateReinforced_C"


def task(width, depth, floors=1, side=0, target=RIP, rate=5):
    return {"target": target, "rate": rate, "width": width, "depth": depth,
            "floors": floors, "side": side}


def make_env(*args, **kwargs):
    env = LayoutEnv()
    env.reset(task(*args, **kwargs))
    return env


def expect_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args} should raise ValueError")


def test_raw_items():
    assert raw_items(build_graph(RIP, 5, alternates=set())) == ["Desc_OreIron_C"]
    stator = build_graph("Desc_Stator_C", 5, alternates=set())
    assert raw_items(stator) == ["Desc_Coal_C", "Desc_OreCopper_C", "Desc_OreIron_C"]
    fake = {"edges": [{"from": "RAW", "to": "R", "item": c} for c in "hgfedcbahd"]
            + [{"from": "R", "to": "OUTPUT", "item": "z"}]}
    assert raw_items(fake) == list("abcdefgh")


def test_edge_cells():
    assert edge_cells(0, 2, 10, 9) == [(0, 3), (0, 6)]
    assert edge_cells(1, 2, 10, 9) == [(3, 0), (6, 0)]
    assert edge_cells(2, 2, 10, 9) == [(9, 3), (9, 6)]
    assert edge_cells(3, 2, 10, 9) == [(3, 8), (6, 8)]
    assert edge_cells(2, 1, 12, 22) == [(11, 11)]
    assert edge_cells(1, 3, 40, 12) == [(10, 0), (20, 0), (30, 0)]
    assert edge_cells(0, 0, 10, 9) == []
    assert edge_cells(1, 2, 11, 5) == [(3, 0), (7, 0)]
    assert edge_cells(0, 2, 5, 11) == [(0, 3), (0, 7)]


def test_reset():
    env = make_env(12, 22)
    assert env.world.owner.shape == (1, 12, 22)
    assert (env.world.owner == EMPTY).all()
    assert env.entries == {"Desc_OreIron_C": (0, 11)}
    assert env.exit == (11, 11)
    assert env.world.reserved.sum() == 2
    assert env.world.reserved[0, 0, 11] and env.world.reserved[0, 11, 11]
    assert [q["recipe"] for q in env.queue] == ["Recipe_IngotIron_C", "Recipe_IronRod_C",
            "Recipe_IronPlate_C", "Recipe_Screw_C", "Recipe_IronPlateReinforced_C"]
    assert not env.done and not env.failed


def test_reset_entries_on_ground_floor_only():
    env = make_env(40, 12, floors=3, side=1, target="Desc_Stator_C")
    assert env.entries == {"Desc_Coal_C": (10, 0), "Desc_OreCopper_C": (20, 0), "Desc_OreIron_C": (30, 0)}
    assert env.exit == (20, 11)
    assert env.world.reserved[0].sum() == 4
    assert not env.world.reserved[1:].any()


def test_reset_again_starts_fresh():
    env = make_env(60, 60)
    env.step(2, 0, 0, 1, 0)
    env.reset(task(30, 40, floors=2, side=3))
    assert env.world.owner.shape == (2, 30, 40)
    assert (env.world.owner == EMPTY).all()
    assert env.world.blocks == []
    assert env.queue[0]["remaining"] == 2
    assert env.exit == (15, 0)
    assert not env.done and not env.failed


def test_reset_too_small():
    env = make_env(8, 8)
    assert env.done and env.failed
    assert not env.n_mask().any()


def test_reset_alternates():
    t = task(60, 60)
    t["alternates"] = {"Recipe_Alternate_Screw_C"}
    env = LayoutEnv()
    env.reset(t)
    assert [q["recipe"] for q in env.queue] == ["Recipe_IngotIron_C", "Recipe_Alternate_Screw_C",
            "Recipe_IronPlate_C", "Recipe_IronPlateReinforced_C"]
    t["alternates"] = {"Recipe_Alternate_ReinforcedIronPlate_2_C"}
    env.reset(t)
    assert "Recipe_IronRod_C" in [q["recipe"] for q in env.queue]
    assert all("Alternate" not in q["recipe"] for q in env.queue)


def test_block():
    env = make_env(12, 22)
    b = env.block(2, 0)
    assert (b["w"], b["d"], b["n"], b["rot"]) == (10, 20, 2, 0)
    assert b["recipe"] == "Recipe_IngotIron_C"
    b = env.block(1, 1)
    assert (b["w"], b["d"], b["rot"]) == (20, 5, 1)
    assert sorted(end_cells(b)) == [(2, 5), (17, -1)]
    assert env.queue[0]["remaining"] == 2


def test_n_limit():
    env = make_env(12, 22)
    assert env.n_limit() == 2
    env = make_env(200, 200, rate=1)
    assert env.n_limit() == 1
    env = make_env(200, 200, rate=30)
    assert env.queue[0]["remaining"] == 12
    assert env.n_limit() == 12
    env = make_env(200, 200, rate=60)
    assert env.queue[0]["remaining"] == 24
    assert env.n_limit() == N_MAX == 16
    env.queue[0] = {"recipe": "Recipe_Rotor_C", "machine": "Build_AssemblerMk1_C",
                    "clock": 1.0, "remaining": 50}
    assert env.n_limit() == 12
    env.queue[0]["remaining"] = 5
    assert env.n_limit() == 5


def test_pos_mask_worked_example():
    env = make_env(12, 22)
    m = env.pos_mask(1)
    assert m.shape == (4, 1, 12, 22)
    assert m.dtype == bool
    assert m[0].sum() == 18 and m[2].sum() == 18
    assert m[1].sum() == 0 and m[3].sum() == 0
    expected = np.zeros((12, 22), dtype=bool)
    expected[1:7, 0:3] = True
    assert (m[0, 0] == expected).all()
    assert (m[2, 0] == expected).all()
    m = env.pos_mask(2)
    assert m.sum() == 6
    assert m[0, 0, 1, 0] and m[0, 0, 1, 2] and not m[0, 0, 2, 0]


def test_pos_mask_rotated_with_entries():
    env = make_env(40, 12, side=1)
    assert env.entries == {"Desc_OreIron_C": (20, 0)}
    m = env.pos_mask(1)
    assert m[0].sum() == 0 and m[2].sum() == 0
    assert m[1].sum() == 124 and m[3].sum() == 124
    assert not m[1, 0, 3, 1] and not m[1, 0, 18, 6]
    assert not m[3, 0, 3, 1] and not m[3, 0, 18, 6]
    assert m[1, 0, 4, 1] and m[1, 0, 18, 5] and m[1, 0, 0, 6] and m[1, 0, 20, 1]
    assert not m[1, 0, 21, 1] and not m[1, 0, 0, 0] and not m[1, 0, 0, 7]


def test_pos_mask_matches_fit_mask():
    env = make_env(50, 45, floors=2, side=2)
    env.step(2, 1, 1, 3, 4)
    for n in range(1, env.n_limit() + 1):
        m = env.pos_mask(n)
        for rot in range(4):
            b = env.block(n, rot)
            fm = env.world.fit_mask(b["w"], b["d"], end_cells(b))
            assert (m[rot, :, :fm.shape[1], :fm.shape[2]] == fm).all()
            assert not m[rot, :, fm.shape[1]:, :].any()
            assert not m[rot, :, :, fm.shape[2]:].any()


def test_pos_mask_does_not_change_state():
    env = make_env(60, 60)
    q = copy.deepcopy(env.queue)
    owner = env.world.owner.copy()
    env.pos_mask(2)
    env.n_mask()
    assert env.queue == q
    assert (env.world.owner == owner).all()
    assert env.world.blocks == []


def test_n_mask_remaining():
    env = make_env(60, 60)
    m = env.n_mask()
    assert m.shape == (N_MAX + 1,)
    assert m.dtype == bool
    assert m.tolist() == [False, True, True] + [False] * (N_MAX - 2)


def test_n_mask_space():
    env = make_env(18, 24, rate=30)
    assert env.queue[0]["remaining"] == 12
    assert env.n_mask().tolist() == [False, True, True, True] + [False] * (N_MAX - 3)
    env = make_env(18, 24, floors=2, rate=30)
    assert env.n_mask().tolist() == [False, True, True, True] + [False] * (N_MAX - 3)


def first_legal(env):
    n = int(np.flatnonzero(env.n_mask())[-1])
    rot, f, x, y = (int(v) for v in np.argwhere(env.pos_mask(n))[0])
    return n, rot, f, x, y


def test_step_worked_example():
    env = make_env(60, 60)
    r = env.step(2, 0, 0, 1, 0)
    assert r == 0.0
    assert not env.done
    assert env.queue[0]["recipe"] == "Recipe_IronRod_C"
    assert (env.world.owner[0, 1:11, 0:20] == 0).all()
    assert (env.world.owner != EMPTY).sum() == 200
    assert env.world.blocks[0]["ends"] == [(0, 2), (11, 17)]
    assert env.world.blocks[0]["block"]["n"] == 2
    r = env.step(1, 1, 0, 20, 30)
    assert r == 0.0
    b = env.world.blocks[1]["block"]
    assert (b["recipe"], b["w"], b["d"], b["rot"]) == ("Recipe_IronRod_C", 20, 9, 1)
    assert (env.world.owner[0, 20:40, 30:39] == 1).all()
    assert env.queue[0]["recipe"] == "Recipe_IronPlate_C"


def test_step_split_recipe():
    env = make_env(60, 60)
    env.step(1, 0, 0, 1, 0)
    assert env.queue[0]["recipe"] == "Recipe_IngotIron_C"
    assert env.queue[0]["remaining"] == 1
    assert env.n_mask().tolist()[:3] == [False, True, False]
    env.step(1, 0, 0, 10, 0)
    assert env.queue[0]["recipe"] == "Recipe_IronRod_C"
    assert [b["block"]["n"] for b in env.world.blocks] == [1, 1]


def test_step_fails_when_nothing_fits():
    env = make_env(12, 22)
    r = env.step(2, 0, 0, 1, 0)
    assert r == -1.0
    assert env.done and env.failed
    assert env.queue[0]["recipe"] == "Recipe_IronRod_C"
    assert not env.n_mask().any()


def test_step_refuses():
    env = make_env(60, 60)
    env.step(1, 0, 0, 1, 0)
    q = copy.deepcopy(env.queue)
    owner = env.world.owner.copy()
    reserved = env.world.reserved.copy()
    bad = [(0, 0, 0, 10, 0), (2, 0, 0, 10, 0), (N_MAX + 1, 0, 0, 10, 0), (1, 4, 0, 10, 0),
           (1, -1, 0, 10, 0), (1, 0, 1, 10, 0), (1, 0, -1, 10, 0), (1, 0, 0, -1, 0),
           (1, 0, 0, 10, -1), (1, 0, 0, 60, 0), (1, 0, 0, 10, 60), (1, 0, 0, 1, 0),
           (1, 0, 0, 5, 0), (1, 0, 0, 0, 30), (1, 0, 0, 55, 0), (1, 0, 0, 10, 41),
           (1, 0, 0, -50, 0), (1, 0, 0, 10, -60), (1, -4, 0, 10, 0), (1, 0, -1, 10, 0)]
    for a in bad:
        expect_error(env.step, *a)
        assert env.queue == q, a
        assert (env.world.owner == owner).all(), a
        assert (env.world.reserved == reserved).all(), a
        assert len(env.world.blocks) == 1, a
        assert not env.done


def test_step_n_max():
    env = make_env(200, 200, rate=60)
    assert env.queue[0]["remaining"] == 24
    expect_error(env.step, N_MAX + 1, 0, 0, 1, 0)
    assert env.world.blocks == [] and env.queue[0]["remaining"] == 24
    env.step(N_MAX, 0, 0, 1, 0)
    assert env.queue[0]["remaining"] == 24 - N_MAX


def test_step_after_done():
    env = make_env(150, 150)
    while not env.done:
        env.step(*first_legal(env))
    assert not env.failed
    expect_error(env.step, 1, 0, 0, 1, 0)
    env = make_env(12, 22)
    env.step(2, 0, 0, 1, 0)
    expect_error(env.step, 1, 0, 0, 1, 0)
    env = make_env(8, 8)
    expect_error(env.step, 1, 0, 0, 1, 0)


def test_greedy_episode_succeeds():
    env = make_env(150, 150, floors=2, rate=20)
    rewards = []
    while not env.done:
        rewards.append(env.step(*first_legal(env)))
    assert rewards[-1] == 1.0 and all(r == 0.0 for r in rewards[:-1])
    assert not env.failed and env.queue == []
    placed = {}
    for b in env.world.blocks:
        placed[b["block"]["recipe"]] = placed.get(b["block"]["recipe"], 0) + b["block"]["n"]
    assert placed == {r: len(nd["clocks"]) for r, nd in env.graph["nodes"].items()}


def slow_pos_mask(env, n):
    F, X, Y = env.world.owner.shape
    blocked = (env.world.owner != EMPTY) | env.world.reserved
    out = np.zeros((4, F, X, Y), dtype=bool)
    for rot in range(4):
        b = env.block(n, rot)
        w, d = b["w"], b["d"]
        for f in range(F):
            for x in range(X - w + 1):
                for y in range(Y - d + 1):
                    ok = not blocked[f, x:x + w, y:y + d].any()
                    for ex, ey in end_cells(b):
                        cx, cy = x + ex, y + ey
                        ok = ok and 0 <= cx < X and 0 <= cy < Y and not blocked[f, cx, cy]
                    out[rot, f, x, y] = ok
    return out


TARGETS = [RIP, "Desc_Stator_C", "Desc_Rotor_C", "Desc_ModularFrame_C", "Desc_IronPlate_C",
           "Desc_Wire_C", "Desc_Cement_C"]


def test_random_agent():
    rng = random.Random(0)
    outcomes = {"ok": 0, "fail": 0}
    for ep in range(25):
        t = task(rng.randint(20, 70), rng.randint(20, 70), floors=rng.randint(1, 3),
                 side=rng.randrange(4), target=rng.choice(TARGETS), rate=rng.choice([2, 5, 10, 20]))
        env = LayoutEnv()
        env.reset(t)
        steps = 0
        while not env.done:
            nm = env.n_mask()
            ns = np.flatnonzero(nm)
            assert len(ns) > 0
            assert ns.tolist() == list(range(1, ns[-1] + 1))
            assert ns[-1] <= env.n_limit()
            n = int(rng.choice(ns))
            pm = env.pos_mask(n)
            if ep < 6:
                assert (pm == slow_pos_mask(env, n)).all()
            cells = np.argwhere(pm)
            a = (n,) + tuple(int(v) for v in cells[rng.randrange(len(cells))])
            r = env.step(*a)
            steps += 1
            assert len(env.world.blocks) == steps
            assert r == (1.0 if env.done and not env.failed else -1.0 if env.failed else 0.0)
        for x, y in list(env.entries.values()) + [env.exit]:
            assert env.world.owner[0, x, y] == EMPTY
        if env.failed:
            outcomes["fail"] += 1
            assert env.queue and not env.n_mask().any()
        else:
            outcomes["ok"] += 1
            assert env.queue == []
    assert outcomes["ok"] > 0 and outcomes["fail"] > 0, outcomes


if __name__ == "__main__":
    tests = [test_raw_items, test_edge_cells, test_reset, test_reset_entries_on_ground_floor_only,
             test_reset_alternates, test_block, test_n_limit, test_pos_mask_worked_example, test_pos_mask_rotated_with_entries,
             test_pos_mask_matches_fit_mask, test_pos_mask_does_not_change_state,
             test_n_mask_remaining, test_n_mask_space, test_reset_too_small, test_step_worked_example,
             test_step_split_recipe, test_step_fails_when_nothing_fits, test_step_refuses,
             test_step_n_max, test_step_after_done, test_reset_again_starts_fresh,
             test_greedy_episode_succeeds, test_random_agent]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")