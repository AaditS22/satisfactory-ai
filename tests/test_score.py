import numpy as np

from layout.env import LayoutEnv
from layout.wire import route_all
from layout.flow import flow
from layout.score import footprint, belt_stats, measure
from layout import score as S


PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0}


def plate_env(steps):
    env = LayoutEnv()
    env.reset(dict(PLATE))
    moves = [(2, 0, 0, 2, 2), (1, 0, 1, 2, 2), (1, 0, 2, 2, 2)]
    for m in moves[:steps]:
        env.step(*m)
    return env


def tiny_routes():
    a = ("A", (0, 0, 1), (1, 2, 2), 10.0)
    b = ("B", (0, 1, 0), (0, 1, 2), 10.0)
    return {"paths": {a: [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 2, 2)],
        b: [(0, 1, 0), (0, 1, 1), (0, 1, 2)]}, "failed": []}


def test_footprint_plate():
    assert footprint(plate_env(0)) == {"bbox": 0, "used": 0, "floors": 0}
    assert footprint(plate_env(2)) == {"bbox": 380, "used": 380, "floors": 2}
    assert footprint(plate_env(3)) == {"bbox": 560, "used": 560, "floors": 3}


def test_footprint_l_shape():
    env = plate_env(0)
    env.world.owner[1, 3:7, 2] = 0
    env.world.owner[1, 3, 2:8] = 1
    assert footprint(env) == {"bbox": 24, "used": 9, "floors": 1}
    env.world.owner[2, 20, 23] = 2
    out = footprint(env)
    assert out == {"bbox": 25, "used": 10, "floors": 2}
    assert all(type(v) is int for v in out.values())


def test_belt_stats_tiny():
    assert belt_stats(tiny_routes()) == {"length": 5, "lifts": 1, "crossings": 1}
    assert belt_stats({"paths": {}, "failed": []}) == {"length": 0, "lifts": 0, "crossings": 0}


def test_belt_stats_turn_is_not_crossing():
    r = tiny_routes()
    c = ("C", (0, 1, 3), (0, 3, 3), 10.0)
    r["paths"][c] = [(0, 1, 3), (0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 3, 3)]
    assert belt_stats(r) == {"length": 9, "lifts": 1, "crossings": 1}
    d = ("D", (2, 5, 0), (0, 5, 0), 10.0)
    r["paths"][d] = [(2, 5, 0), (2, 5, 1), (0, 5, 1), (0, 5, 0)]
    assert belt_stats(r) == {"length": 11, "lifts": 2, "crossings": 1}
    e = ("E", (0, 0, 5), (0, 2, 5), 10.0)
    f = ("F", (0, 1, 6), (0, 0, 4), 10.0)
    r["paths"][e] = [(0, 0, 5), (0, 1, 5), (0, 2, 5)]
    r["paths"][f] = [(0, 1, 6), (0, 1, 5), (0, 0, 5), (0, 0, 4)]
    assert belt_stats(r) == {"length": 16, "lifts": 2, "crossings": 1}


def test_belt_stats_plate():
    env = plate_env(3)
    assert belt_stats(route_all(env)) == {"length": 64, "lifts": 4, "crossings": 0}


def test_measure_plate():
    env = plate_env(3)
    routes = route_all(env)
    out = measure(env, routes, flow(env, routes))
    assert out == {"links": 5, "routed": 1.0, "fraction": 1.0, "passed": True, "length": 64,
        "lifts": 4, "crossings": 0, "bbox": 560, "used": 560, "floors": 3}


def test_measure_failed_link():
    env = plate_env(3)
    routes = route_all(env)
    ore = [L for L in routes["paths"] if L[0] == "Desc_OreIron_C"][0]
    paths = dict(routes["paths"])
    del paths[ore]
    broken = {"paths": paths, "failed": [ore]}
    out = measure(env, broken, flow(env, broken))
    assert out["links"] == 5
    assert abs(out["routed"] - 0.8) < 1e-9
    assert out["fraction"] == 0 and out["passed"] is False
    assert out["length"] == 55 and out["lifts"] == 4
    assert out["bbox"] == 560


def slow_footprint(env):
    occ = env.world.owner >= 0
    bbox = used = floors = 0
    for f in range(occ.shape[0]):
        cells = [(x, y) for x in range(occ.shape[1]) for y in range(occ.shape[2]) if occ[f, x, y]]
        if cells:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            bbox += (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
            used += len(cells)
            floors += 1
    return {"bbox": bbox, "used": used, "floors": floors}


def test_random_vs_view():
    from layout.view import random_episode, route_info
    checked = 0
    for seed in [0, 2, 3, 5, 7, 11]:
        env = random_episode(seed)
        assert footprint(env) == slow_footprint(env)
        if env.failed:
            continue
        routes = route_all(env)
        info = route_info(env, routes)
        got = belt_stats(routes)
        assert got == {"length": info["length"], "lifts": info["n_lifts"], "crossings": len(info["crossings"])}
        out = measure(env, routes, flow(env, routes))
        assert out["passed"] == (len(routes["failed"]) == 0)
        assert out["used"] == sum(b["block"]["w"] * b["block"]["d"] for b in env.world.blocks)
        assert out["used"] <= out["bbox"]
        checked += 1
    assert checked >= 4


PLATE_TERMS = {"links": 5, "routed": 1.0, "fraction": 1.0, "passed": True, "length": 64,
    "lifts": 4, "crossings": 0, "bbox": 560, "used": 560, "floors": 3}


def close(a, b):
    return abs(a - b) < 1e-9


def test_constants():
    assert set(S.WEIGHTS) == {"p_belt", "p_lifts", "p_crossings", "p_sprawl", "p_floors"}
    assert close(sum(S.WEIGHTS.values()), 1.0)
    assert (S.BELT_REF, S.LIFT_REF, S.CROSS_REF, S.SPRAWL_REF, S.FLOOR_REF) == (32, 1.0, 0.25, 0.5, 2)


def test_parts_plate():
    p = S.parts(dict(PLATE_TERMS))
    assert set(p) == set(S.WEIGHTS)
    want = {"p_belt": 0.4, "p_lifts": 0.8, "p_crossings": 0.0, "p_sprawl": 0.0, "p_floors": 1.0}
    assert all(close(p[k], want[k]) for k in want)


def test_parts_other():
    t = {"links": 8, "length": 512, "lifts": 2, "crossings": 6, "bbox": 900, "used": 600, "floors": 2}
    p = S.parts(t)
    want = {"p_belt": 2.0, "p_lifts": 0.25, "p_crossings": 3.0, "p_sprawl": 1.0, "p_floors": 0.5}
    assert all(close(p[k], want[k]) for k in want)
    assert t == {"links": 8, "length": 512, "lifts": 2, "crossings": 6, "bbox": 900, "used": 600, "floors": 2}


def test_penalty_and_score():
    assert close(S.penalty(dict(PLATE_TERMS)), 0.41)
    assert close(S.score(dict(PLATE_TERMS)), 1 / 1.41)
    t = {"links": 8, "length": 512, "lifts": 2, "crossings": 6, "bbox": 900, "used": 600, "floors": 2,
        "passed": True, "fraction": 1.0, "routed": 1.0}
    assert close(S.penalty(t), 0.7 + 0.0375 + 0.45 + 0.2 + 0.075)
    assert close(S.score(t), 1 / (1 + 1.4625))
    perfect = {"links": 3, "length": 0, "lifts": 0, "crossings": 0, "bbox": 100, "used": 100, "floors": 1,
        "passed": True, "fraction": 1.0, "routed": 1.0}
    assert close(S.score(perfect), 1.0)


def test_score_broken():
    t = dict(PLATE_TERMS, passed=False, fraction=0.0, routed=0.8, length=55)
    assert close(S.score(t), -0.6)
    t = dict(PLATE_TERMS, passed=False, fraction=0.5, routed=0.9)
    assert close(S.score(t), -0.3)
    t = dict(PLATE_TERMS, passed=False, fraction=0.0, routed=0.0)
    assert close(S.score(t), -1.0)


def test_placed_fraction():
    assert close(S.placed_fraction(plate_env(0)), 0.0)
    assert close(S.placed_fraction(plate_env(2)), 0.75)
    assert close(S.placed_fraction(plate_env(3)), 1.0)


def test_score_episode_plate():
    s, t = S.score_episode(plate_env(3))
    assert close(s, 1 / 1.41)
    for k, v in PLATE_TERMS.items():
        assert t[k] == v, k
    assert close(t["p_belt"], 0.4) and close(t["p_floors"], 1.0)
    assert close(t["penalty"], 0.41) and t["placed"] == 1.0
    try:
        S.score_episode(plate_env(2))
        assert False
    except ValueError:
        pass


def test_score_episode_unfinished():
    env = LayoutEnv()
    env.reset(dict(PLATE, floors=1))
    env.step(2, 0, 0, 7, 2)
    assert env.done and env.failed
    s, t = S.score_episode(env)
    assert close(s, -1.5)
    assert t == {"placed": 0.5}


def test_score_episode_tiers_random():
    from layout.view import random_episode
    seen = set()
    for seed in [0, 1, 2, 3, 5, 7, 11, 26]:
        env = random_episode(seed)
        s, t = S.score_episode(env)
        if env.failed:
            assert -2 <= s < -1
            assert close(s, -2 + S.placed_fraction(env))
            seen.add("unfinished")
        elif t["passed"]:
            assert 0 < s <= 1
            assert close(s, 1 / (1 + t["penalty"]))
            assert close(t["penalty"], sum(S.WEIGHTS[k] * t[k] for k in S.WEIGHTS))
            assert t["lifts"] == int(t["lifts"]) and t["floors"] in (1, 2, 3)
            seen.add("valid")
        else:
            assert -1 <= s < 0
            seen.add("broken")
    assert {"valid", "broken", "unfinished"} <= seen


TESTS = [test_footprint_plate, test_footprint_l_shape, test_belt_stats_tiny, test_belt_stats_turn_is_not_crossing,
    test_belt_stats_plate, test_measure_plate, test_measure_failed_link, test_random_vs_view,
    test_constants, test_parts_plate, test_parts_other, test_penalty_and_score, test_score_broken,
    test_placed_fraction, test_score_episode_plate, test_score_episode_unfinished, test_score_episode_tiers_random]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")