import math

import numpy as np

import layout.anneal as A
from layout.api import evaluate
from layout.baseline import baseline_a
from layout.anneal import anneal, baseline_b
from layout.anneal import (temperature, accept, shift, rotate, swap, resplit, neighbour,
                           T0, T1, STEP)


def plate_plan():
    return [[2, 0, 0, 3, 22], [2, 0, 0, 14, 28]]


def test_temperature_hand():
    assert abs(temperature(0, 101) - 0.02) < 1e-12
    assert abs(temperature(100, 101) - 0.0005) < 1e-12
    assert abs(temperature(50, 101) - 0.0031622776601683794) < 1e-12
    assert abs(temperature(1, 3, 1.0, 0.25) - 0.5) < 1e-12
    assert temperature(0, 1) == T0
    assert T0 == 0.02 and T1 == 0.0005


def test_temperature_goes_down():
    ts = [temperature(i, 50) for i in range(50)]
    assert all(a > b for a, b in zip(ts, ts[1:]))


def test_accept_hand():
    assert accept(0.70, 0.71, 0.01, 0.99)
    assert accept(0.70, 0.70, 0.01, 0.99)
    assert accept(0.70, 0.69, 0.01, 0.30)
    assert not accept(0.70, 0.69, 0.01, 0.40)
    assert accept(0.70, 0.68, 0.01, math.exp(-2) - 1e-9)
    assert not accept(0.70, 0.68, 0.01, math.exp(-2) + 1e-9)


def test_accept_tiny_temperature():
    assert not accept(0.7, 0.6, 1e-6, 0.0)
    assert accept(0.6, 0.7, 1e-6, 0.999)
    assert not accept(0.5, -1.4, 0.0005, 0.0)


def test_shift_hand():
    p = plate_plan()
    q = shift(p, 1, -2, 3)
    assert q == [[2, 0, 0, 3, 22], [2, 0, 0, 12, 31]]
    assert p == plate_plan()
    assert shift(p, 0, 0, -5) == [[2, 0, 0, 3, 17], [2, 0, 0, 14, 28]]


def test_rotate_hand():
    p = plate_plan()
    q = rotate(p, 0, 3)
    assert q == [[2, 3, 0, 3, 22], [2, 0, 0, 14, 28]]
    assert rotate(q, 0, 2) == [[2, 1, 0, 3, 22], [2, 0, 0, 14, 28]]
    assert p == plate_plan()


def test_swap_hand():
    p = [[2, 0, 0, 3, 22], [1, 1, 1, 14, 28], [3, 2, 2, 5, 5]]
    q = swap(p, 0, 1)
    assert q == [[2, 0, 1, 14, 28], [1, 1, 0, 3, 22], [3, 2, 2, 5, 5]]
    assert swap(p, 2, 0) == [[2, 0, 2, 5, 5], [1, 1, 1, 14, 28], [3, 2, 0, 3, 22]]
    assert p == [[2, 0, 0, 3, 22], [1, 1, 1, 14, 28], [3, 2, 2, 5, 5]]


def test_resplit_hand():
    p = [[3, 0, 0, 1, 1], [2, 1, 0, 9, 9]]
    assert resplit(p, 0, 1) == [[2, 0, 0, 1, 1], [3, 1, 0, 9, 9]]
    assert resplit(p, 1, 0) == [[4, 0, 0, 1, 1], [1, 1, 0, 9, 9]]
    assert p == [[3, 0, 0, 1, 1], [2, 1, 0, 9, 9]]


def test_resplit_removes_empty_block():
    p = [[1, 0, 0, 1, 1], [2, 1, 0, 9, 9], [4, 2, 1, 3, 3]]
    assert resplit(p, 0, 1) == [[3, 1, 0, 9, 9], [4, 2, 1, 3, 3]]
    q = [[2, 0, 0, 1, 1], [1, 1, 0, 9, 9], [4, 2, 1, 3, 3]]
    assert resplit(q, 1, 0) == [[3, 0, 0, 1, 1], [4, 2, 1, 3, 3]]
    assert p == [[1, 0, 0, 1, 1], [2, 1, 0, 9, 9], [4, 2, 1, 3, 3]]


def kind_of(old, new):
    if len(new) != len(old) or any(a[0] != b[0] for a, b in zip(old, new)):
        return "resplit"
    diff = [i for i in range(len(old)) if old[i] != new[i]]
    if len(diff) == 2:
        return "swap"
    assert len(diff) == 1
    a, b = old[diff[0]], new[diff[0]]
    if a[1] != b[1]:
        assert a[2:] == b[2:]
        return "rotate"
    return "shift"


def test_neighbour_invariants():
    plan = [[3, 0, 0, 2, 2], [2, 0, 0, 20, 2], [1, 1, 1, 5, 30]]
    recipes = ["R_a", "R_a", "R_b"]
    keep = [list(m) for m in plan]
    seen = set()
    grew = set()
    for seed in range(400):
        new = neighbour(plan, recipes, np.random.default_rng(seed))
        assert plan == keep
        if new is None:
            continue
        assert new is not plan and new != plan
        assert sum(m[0] for m in new) == 6
        assert all(isinstance(v, int) for m in new for v in m)
        kind = kind_of(plan, new)
        seen.add(kind)
        if kind == "resplit":
            assert new[-1] == plan[-1]
            assert sum(m[0] for m in new[:-1]) == 5
            grew.add(new[0][0] > plan[0][0])
        if kind == "shift":
            i = [k for k in range(3) if plan[k] != new[k]][0]
            dx, dy = new[i][3] - plan[i][3], new[i][4] - plan[i][4]
            assert max(abs(dx), abs(dy)) <= STEP and (dx, dy) != (0, 0)
            assert new[i][:3] == plan[i][:3]
        if kind == "swap":
            i, j = [k for k in range(3) if plan[k] != new[k]]
            assert new[i][:2] == plan[i][:2] and new[i][2:] == plan[j][2:]
        if kind == "rotate":
            i = [k for k in range(3) if plan[k] != new[k]][0]
            assert 0 <= new[i][1] < 4
    assert seen == {"shift", "rotate", "swap", "resplit"}
    assert grew == {True, False}


def test_neighbour_small_plans():
    rng = np.random.default_rng(0)
    assert neighbour([], [], rng) is None
    for seed in range(100):
        new = neighbour([[2, 0, 0, 3, 3]], ["R_a"], np.random.default_rng(seed))
        assert new is None or kind_of([[2, 0, 0, 3, 3]], new) in ("shift", "rotate")


PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 34, "depth": 48,
         "floors": 1, "side": 0, "alternates": set()}
START = [[2, 0, 0, 3, 22], [2, 0, 0, 14, 28]]
BETTER = [[2, 0, 0, 3, 22], [2, 0, 0, 14, 27]]
WORSE = [[2, 0, 0, 2, 2], [2, 0, 0, 14, 17]]
OUTSIDE = [[2, 0, 0, 40, 0], [2, 0, 0, 14, 28]]
UNFINISHED = [[2, 0, 0, 3, 22]]
SPLIT3 = [[1, 0, 0, 3, 22], [1, 0, 0, 3, 0], [2, 0, 0, 14, 28]]
RECIPES = ["Recipe_IngotIron_C", "Recipe_IronPlate_C"]


def scripted(script, seen):
    def fake(plan, recipes, rng):
        seen.append(([list(m) for m in plan], list(recipes)))
        nxt = script.pop(0)
        return None if nxt is None else [list(m) for m in nxt]
    return fake


def run_scripted(script, accept_fn, iters):
    seen, calls = [], []
    old_n, old_a = A.neighbour, A.accept

    def acc(old, new, t, u):
        calls.append((old, new, t, u))
        return accept_fn(old, new, t, u)

    A.neighbour, A.accept = scripted(list(script), seen), acc
    try:
        res = anneal(PLATE, [list(m) for m in START], iters, np.random.default_rng(0))
    finally:
        A.neighbour, A.accept = old_n, old_a
    return res, seen, calls


def test_anneal_keeps_best_not_current():
    res, seen, calls = run_scripted([None, OUTSIDE, UNFINISHED, BETTER, WORSE], lambda o, n, t, u: True, 5)
    assert res["plan"] == BETTER
    assert abs(res["score"] - 0.8076437714079683) < 1e-12
    assert abs(evaluate(PLATE, res["plan"])["score"] - res["score"]) < 1e-12
    assert set(res) >= {"env", "routes", "score", "terms", "plan"}
    assert [p for p, r in seen] == [START, START, START, START, BETTER]
    assert all(r == RECIPES for p, r in seen)


def test_anneal_accept_arguments():
    res, seen, calls = run_scripted([None, OUTSIDE, UNFINISHED, BETTER, WORSE], lambda o, n, t, u: True, 5)
    assert len(calls) == 2
    (o1, n1, t1, u1), (o2, n2, t2, u2) = calls
    assert abs(o1 - 0.7943544097308415) < 1e-12 and abs(n1 - 0.8076437714079683) < 1e-12
    assert abs(o2 - 0.8076437714079683) < 1e-12 and abs(n2 - 0.6801275239107333) < 1e-12
    assert abs(t1 - temperature(3, 5)) < 1e-15 and abs(t2 - temperature(4, 5)) < 1e-15
    assert 0 <= u1 < 1 and 0 <= u2 < 1


def test_anneal_rejected_change_is_forgotten():
    res, seen, calls = run_scripted([WORSE, BETTER, None], lambda o, n, t, u: n >= o, 3)
    assert [p for p, r in seen] == [START, START, BETTER]
    assert res["plan"] == BETTER


def test_anneal_nothing_accepted():
    plan = [list(m) for m in START]
    res, seen, calls = run_scripted([BETTER, WORSE, BETTER], lambda o, n, t, u: False, 3)
    assert res["plan"] == START
    assert abs(res["score"] - 0.7943544097308415) < 1e-12
    assert [p for p, r in seen] == [START, START, START]


def test_anneal_zero_iters_and_no_mutation():
    plan = [list(m) for m in START]
    res = anneal(PLATE, plan, 0, np.random.default_rng(0))
    assert res["plan"] == START and plan == START
    assert abs(res["score"] - 0.7943544097308415) < 1e-12
    res = anneal(PLATE, plan, 25, np.random.default_rng(1))
    assert plan == START


def test_anneal_recipes_follow_current_plan():
    res, seen, calls = run_scripted([SPLIT3, None], lambda o, n, t, u: True, 2)
    assert [p for p, r in seen] == [START, SPLIT3]
    assert seen[1][1] == ["Recipe_IngotIron_C", "Recipe_IngotIron_C", "Recipe_IronPlate_C"]
    assert res["plan"] == START


def test_anneal_other_errors_surface():
    bad = [[2, 0, 0, "x", 22], [2, 0, 0, 14, 28]]
    try:
        run_scripted([bad], lambda o, n, t, u: True, 1)
    except TypeError:
        return
    assert False


def test_anneal_real_runs():
    a = anneal(PLATE, START, 40, np.random.default_rng(5))
    b = anneal(PLATE, START, 40, np.random.default_rng(5))
    assert a["plan"] == b["plan"] and a["score"] == b["score"]
    assert a["score"] >= 0.7943544097308415
    assert abs(evaluate(PLATE, a["plan"])["score"] - a["score"]) < 1e-12
    assert all(isinstance(v, int) for m in a["plan"] for v in m)


def test_baseline_b_random_tasks():
    from layout.tasks import random_task
    for seed in (0, 4):
        task = random_task(np.random.default_rng(seed))
        res = baseline_b(task, 30, seed=1)
        start = baseline_a(task)
        assert res["score"] >= start["score"]
        assert abs(evaluate(task, res["plan"])["score"] - res["score"]) < 1e-12
        again = baseline_b(task, 30, seed=1)
        assert again["plan"] == res["plan"]


TESTS = [test_temperature_hand, test_temperature_goes_down, test_accept_hand, test_accept_tiny_temperature,
         test_shift_hand, test_rotate_hand, test_swap_hand, test_resplit_hand, test_resplit_removes_empty_block,
         test_neighbour_invariants, test_neighbour_small_plans,
         test_anneal_keeps_best_not_current, test_anneal_accept_arguments, test_anneal_rejected_change_is_forgotten,
         test_anneal_nothing_accepted, test_anneal_zero_iters_and_no_mutation, test_anneal_recipes_follow_current_plan,
         test_anneal_other_errors_surface, test_anneal_real_runs,
         test_baseline_b_random_tasks]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(TESTS)} passed")