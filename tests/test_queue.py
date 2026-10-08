"""Tests for layout/queue.py (the recipe queue for block placement).

Run from the repo root:  python -m tests.test_queue
"""
import math

from layout.blocks import make_block
from layout.queue import depths, make_queue, max_block, take
from solver.graph import build_graph, pick_belt

INGOT, ROD, SCREW, ROTOR = "Recipe_IngotIron_C", "Recipe_IronRod_C", "Recipe_Screw_C", "Recipe_Rotor_C"
PLATE, RIP, FRAME = "Recipe_IronPlate_C", "Recipe_IronPlateReinforced_C", "Recipe_ModularFrame_C"
TARGETS = [("Desc_Rotor_C", 10), ("Desc_ModularFrame_C", 5), ("Desc_Motor_C", 5),
           ("Desc_SteelPlate_C", 30), ("Desc_Stator_C", 7)]


def rotor_graph():
    return build_graph("Desc_Rotor_C", 10, alternates=set())


def frame_graph():
    return build_graph("Desc_ModularFrame_C", 5, alternates=set())


# ---------------------------------------------------------------- depths

def test_depths():
    # rod feeds rotor directly (1 step) and via screws (2 steps): the longest path counts
    assert depths(rotor_graph()) == {INGOT: 3, ROD: 2, SCREW: 1, ROTOR: 0}
    assert depths(frame_graph()) == {INGOT: 4, ROD: 3, PLATE: 2, SCREW: 2, RIP: 1, FRAME: 0}


# ---------------------------------------------------------------- make_queue

def test_make_queue_rotor():
    q = make_queue(rotor_graph())
    assert [e["recipe"] for e in q] == [INGOT, ROD, SCREW, ROTOR]
    assert [e["remaining"] for e in q] == [4, 8, 7, 3]
    assert [e["machine"] for e in q] == ["Build_SmelterMk1_C", "Build_ConstructorMk1_C",
                                         "Build_ConstructorMk1_C", "Build_AssemblerMk1_C"]
    for e, clock in zip(q, [0.9375, 0.9375, 25 / 28, 5 / 6]):
        assert math.isclose(e["clock"], clock), (e["recipe"], e["clock"])
    assert set(q[0]) == {"recipe", "machine", "clock", "remaining"}


def test_make_queue_ties_and_order():
    # plate and screw both have depth 2: ties break alphabetically by recipe id
    q = make_queue(frame_graph())
    assert [e["recipe"] for e in q] == [INGOT, ROD, PLATE, SCREW, RIP, FRAME]
    for target, rate in TARGETS:
        g = build_graph(target, rate, alternates=set())
        q = make_queue(g)
        order = [e["recipe"] for e in q]
        assert sorted(order) == sorted(g["nodes"]), target
        assert sum(e["remaining"] for e in q) == sum(len(n["clocks"]) for n in g["nodes"].values())
        # every recipe comes after all the recipes that supply it
        for edge in g["edges"]:
            if edge["from"] != "RAW" and edge["to"] != "OUTPUT":
                assert order.index(edge["from"]) < order.index(edge["to"]), (target, edge)


def test_make_queue_copies():
    # taking from the queue must not change the solver's graph
    g = rotor_graph()
    q = make_queue(g)
    take(q, 4)
    assert len(g["nodes"][INGOT]["clocks"]) == 4


# ---------------------------------------------------------------- max_block

def test_max_block_rotor():
    q = make_queue(rotor_graph())
    # Mk6 = 1200/min: ingot 1200 / (30 x 0.9375) = 42.7 -> 42, ...
    assert [max_block(e) for e in q] == [42, 85, 33, 14]
    # Mk3 = 270/min: rotor 270 / (100 screws x 5/6) = 3.24 -> 3
    assert [max_block(e, 3) for e in q] == [9, 19, 7, 3]


def test_max_block_zero():
    # one rotor assembler needs 100 x 5/6 = 83.3 screws/min: more than a Mk1 belt (60)
    rotor = make_queue(rotor_graph())[-1]
    assert max_block(rotor, 1) == 0
    assert max_block(rotor, 2) == 1


def test_max_block_fits_belts():
    # a block of max_block machines fits one belt on every row; one more does not
    for target, rate in TARGETS:
        for e in make_queue(build_graph(target, rate, alternates=set())):
            for tier in (1, 3, 6):
                n = max_block(e, tier)
                assert n >= 0, (e, tier)
                b = make_block(e["recipe"], max(n, 1), e["clock"])
                rates = [i["rate"] for i in b["inputs"]] + [b["output"]["rate"]]
                if n == 0:   # even one machine overflows this tier
                    assert any(pick_belt(r, tier)[1] > 1 for r in rates), (e["recipe"], tier)
                    continue
                b = make_block(e["recipe"], n, e["clock"])
                rates = [i["rate"] for i in b["inputs"]] + [b["output"]["rate"]]
                assert all(pick_belt(r, tier)[1] == 1 for r in rates), (e["recipe"], tier, n)
                b = make_block(e["recipe"], n + 1, e["clock"])
                rates = [i["rate"] for i in b["inputs"]] + [b["output"]["rate"]]
                assert any(pick_belt(r, tier)[1] > 1 for r in rates), (e["recipe"], tier, n)


# ---------------------------------------------------------------- take

def test_take():
    q = make_queue(rotor_graph())
    r, n, clock = take(q, 2)
    assert (r, n) == (INGOT, 2) and math.isclose(clock, 0.9375)
    assert q[0]["recipe"] == INGOT and q[0]["remaining"] == 2
    assert take(q, 2)[:2] == (INGOT, 2)
    assert q[0]["recipe"] == ROD and len(q) == 3       # ingots used up -> removed

    for bad in (0, -1, 9):                             # 8 rods left, so 9 is too many
        try:
            take(q, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"take({bad}) should raise ValueError")
    assert q[0]["remaining"] == 8                      # a refused take changes nothing


def test_take_respects_belt_limit():
    q = make_queue(rotor_graph())
    take(q, 4)
    take(q, 8)
    take(q, 7)
    try:
        take(q, 3, max_tier=1)    # 3 rotor assemblers need 250 screws/min > Mk1
    except ValueError:
        pass
    else:
        raise AssertionError("take should refuse a block that overflows the belt")
    assert take(q, 3, max_tier=3)[:2] == (ROTOR, 3)
    assert q == []


def test_drain():
    # placing max-size blocks until empty uses every machine exactly once
    for target, rate in TARGETS:
        g = build_graph(target, rate, alternates=set())
        q = make_queue(g)
        placed = {}
        while q:
            n = min(q[0]["remaining"], max_block(q[0], 3))
            r, n, _ = take(q, n)
            placed[r] = placed.get(r, 0) + n
        assert placed == {r: len(node["clocks"]) for r, node in g["nodes"].items()}, target


if __name__ == "__main__":
    tests = [test_depths, test_make_queue_rotor, test_make_queue_ties_and_order, test_make_queue_copies,
             test_max_block_rotor, test_max_block_zero, test_max_block_fits_belts, test_take, test_take_respects_belt_limit,
             test_drain]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")