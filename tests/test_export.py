import math

import numpy as np

from executor.plan import validate_plan
from layout.blocks import CATALOG
from layout.env import LayoutEnv
from layout.tasks import random_task
from layout.wire import route_all
from layout.api import load, evaluate, play
from layout import export as X


HAND = {"target": "Desc_IronPlate_C", "rate": 30, "width": 34, "depth": 48, "floors": 1, "side": 0}
HAND_PLAN = [[2, 0, 0, 2, 2], [2, 0, 0, 14, 17]]


def hand():
    res = evaluate(dict(HAND), HAND_PLAN)
    return res, X.layout_plan(res["env"], res["routes"], "plate_hand")


def one_floor_episode(seed):
    rng = np.random.default_rng(seed)
    t = random_task(rng)
    t["floors"] = 1
    t["width"] = min(96, int(t["width"] * 1.7))
    t["depth"] = min(96, int(t["depth"] * 1.7))
    env = LayoutEnv()
    env.reset(t)
    while not env.done:
        ns = np.flatnonzero(env.n_mask())
        n = int(ns[rng.integers(len(ns))])
        spots = np.argwhere(env.pos_mask(n))
        env.step(n, *(int(v) for v in spots[rng.integers(len(spots))]))
    return env


def check_belts(plan):
    ents = {e["id"]: e for e in plan["entities"]}
    used = set()
    for e in plan["entities"]:
        if e["kind"] != "belt":
            continue
        a_ent, b_ent = ents[e["from"]], ents[e["to"]]
        dirs = {p["name"]: p["dir"] for p in CATALOG[a_ent["class"]]["ports"]}
        assert dirs[e["from_port"]] == "out", e
        dirs = {p["name"]: p["dir"] for p in CATALOG[b_ent["class"]]["ports"]}
        assert dirs[e["to_port"]] == "in", e
        for key in [(e["from"], e["from_port"]), (e["to"], e["to_port"])]:
            assert key not in used, key
            used.add(key)
        a = X.port_pos(a_ent, e["from_port"])
        b = X.port_pos(b_ent, e["to_port"])
        assert abs(a[2] - b[2]) < 1e-9
        if a_ent["kind"] == "machine" or b_ent["kind"] == "machine":
            continue
        assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6, (e, a, b)
        assert math.dist(a, b) >= X.MIN_BELT - 1e-9
    return used


def test_turn_vec_and_ports():
    assert X.turn_vec((1, 0), 90) == (0, 1)
    assert X.turn_vec((0, 1), 270) == (1, 0)
    assert X.turn_vec((1, 0), 180) == (-1, 0)
    assert X.port_with(X.MERGER, 0, (0, 1), "in") == "Input2"
    assert X.port_with(X.MERGER, 90, (1, 0), "in") == "Input3"
    assert X.port_with(X.SPLITTER, 270, (0, -1), "out") == "Output1"
    assert X.port_with(X.SPLITTER, 180, (0, 1), "out") == "Output3"
    f = X.port_facing(X.SPLITTER, 90)
    assert f == {"Output1": (0, 1), "Input1": (0, -1), "Output3": (1, 0), "Output2": (-1, 0)}
    try:
        X.port_with(X.MERGER, 0, (1, 0), "in")
        assert False
    except ValueError:
        pass
    e = {"kind": "attachment", "class": X.MERGER, "pos": [0.5, 12.5, 1.0], "yaw": 0}
    assert X.port_pos(e, "Input2") == (0.5, 13.5, 1.0)
    c = {"kind": "container", "class": X.CONTAINER, "pos": [-7.5, 24.5, 0.0], "yaw": 270}
    assert X.port_pos(c, "Output1") == (-3.5, 24.5, 1.0)


def test_side_of():
    for side in range(4):
        env = play(dict(HAND, side=side), [])
        assert X._side_of(env) == side
        env = play(dict(HAND, side=side, width=40, depth=24), [])
        assert X._side_of(env) == side


def test_floor_foundations():
    ents = [{"id": "a", "kind": "machine", "pos": [4.0, 4.0, 0.0]},
        {"id": "b", "kind": "attachment", "pos": [10.5, 3.5, 17.0]}, {"id": "c", "kind": "belt"}]
    out = X.floor_foundations(ents)
    f0 = [e for e in out if e["id"].startswith("f0_")]
    f1 = [e for e in out if e["id"].startswith("f1_")]
    assert len(f0) + len(f1) == len(out)
    assert all(e["pos"][2] == -0.5 for e in f0) and all(e["pos"][2] == 15.5 for e in f1)
    assert {(e["pos"][0], e["pos"][1]) for e in f0} == {(-4.0, -4.0), (-4.0, 4.0), (-4.0, 12.0), (4.0, -4.0),
        (4.0, 4.0), (4.0, 12.0), (12.0, -4.0), (12.0, 4.0), (12.0, 12.0)}
    assert len(f1) == 9 and len({e["id"] for e in out}) == len(out)


def test_hand_plan():
    res, plan = hand()
    assert validate_plan(plan) == []
    kinds = {}
    for e in plan["entities"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    assert kinds["machine"] == 4 and kinds["attachment"] == 12 and kinds["container"] == 2 and kinds["belt"] == 19
    ents = {e["id"]: e for e in plan["entities"]}
    want = {"j1": ([32.5, 34.5, 1.0], 270), "j2": ([32.5, 24.5, 1.0], 0), "j3": ([0.5, 24.5, 1.0], 270),
        "j4": ([0.5, 4.5, 1.0], 0)}
    for k, (pos, yaw) in want.items():
        assert ents[k]["pos"] == pos and ents[k]["yaw"] == yaw and ents[k]["class"] == X.MERGER, k
    assert ents["in_Desc_OreIron_C"]["pos"] == [-7.5, 24.5, 0.0] and ents["in_Desc_OreIron_C"]["yaw"] == 270
    assert ents["in_Desc_OreIron_C"]["fill"] == {"item": "Desc_OreIron_C", "amount": X.FILL}
    assert ents["out"]["pos"] == [41.5, 24.5, 0.0] and "fill" not in ents["out"]
    links = {(e["from"], e["to"]) for e in plan["entities"] if e["kind"] == "belt"}
    assert {("in_Desc_OreIron_C", "j3"), ("j3", "j4"), ("j4", "b0_s0_0"), ("b0_g1", "b1_s0_0"), ("b1_g1", "j1"),
        ("j1", "j2"), ("j2", "out")} <= links


def test_hand_belts():
    res, plan = hand()
    used = check_belts(plan)
    assert ("j3", "Input3") in used and ("j4", "Input2") in used
    assert ("j1", "Input3") in used and ("j2", "Input2") in used


def test_long_runs_split():
    keep = X.MAX_RUN
    X.MAX_RUN = 5
    try:
        res, plan = hand()
    finally:
        X.MAX_RUN = keep
    used = check_belts(plan)
    straight = [e for e in plan["entities"] if e["id"].startswith("j") and (e["id"], "Input1") in used]
    assert len(straight) >= 4
    assert sum(1 for e in plan["entities"] if e["id"].startswith("j")) > 4


def test_hand_file():
    task, plan = load("tests/layouts/plate_hand.json")
    assert task == HAND and plan == HAND_PLAN


def test_junction_layouts():
    total = {"split": 0, "merge": 0}
    for seed in [4, 9, 16]:
        env = one_floor_episode(seed)
        routes = route_all(env)
        plan = X.layout_plan(env, routes, f"s{seed}")
        assert validate_plan(plan) == []
        used = check_belts(plan)
        ents = {e["id"]: e for e in plan["entities"]}
        pieces = [e for e in plan["entities"] if e["id"].startswith("j")]
        for p in pieces:
            ins = [k for k in used if k[0] == p["id"] and k[1].startswith("Input")]
            outs = [k for k in used if k[0] == p["id"] and k[1].startswith("Output")]
            if p["class"] == X.SPLITTER:
                assert (len(ins), len(outs)) == (1, 2)
                total["split"] += 1
            else:
                assert len(outs) == 1 and len(ins) in (1, 2)
                total["merge"] += len(ins) == 2
        machines = sum(b["block"]["n"] for b in env.world.blocks)
        assert sum(1 for e in plan["entities"] if e["kind"] == "machine") == machines
        assert sum(1 for e in plan["entities"] if e["kind"] == "container") == len(env.entries) + 1
    assert total["split"] >= 2 and total["merge"] >= 1


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_raises():
    multi = evaluate({"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0},
        [[2, 0, 0, 2, 2], [1, 0, 1, 2, 2], [1, 0, 2, 2, 2]])
    assert raises(lambda: X.layout_plan(multi["env"], multi["routes"], "m"))
    res, plan = hand()
    part = play(dict(HAND), HAND_PLAN[:1])
    assert raises(lambda: X.layout_plan(part, res["routes"], "p"))
    one = next(iter(res["routes"]["paths"]))
    broken = {"paths": {k: v for k, v in res["routes"]["paths"].items() if k != one}, "failed": [one]}
    assert raises(lambda: X.layout_plan(res["env"], broken, "b"))


TESTS = [test_turn_vec_and_ports, test_side_of, test_floor_foundations, test_hand_plan, test_hand_belts,
    test_long_runs_split, test_hand_file, test_junction_layouts, test_raises]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")