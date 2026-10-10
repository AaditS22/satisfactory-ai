import numpy as np

from solver.graph import build_graph
from solver.lp_solve import RESOURCE_LIMITS, recipe_by_id, solve_lp
from siting.choose import min_floor_area
from layout.env import LayoutEnv
from layout.tasks import (FLOORS, GRID_MAX, MAX_MACHINES, RATES, STEP, all_alternates, machine_count,
                          min_sides, random_alternates, random_task, round_up, scale, site_size, targets,
                          valid_rates)

RIP = "Desc_IronPlateReinforced_C"
KEYS = {"target", "rate", "width", "depth", "floors", "side", "alternates"}


def test_round_up():
    assert round_up(1) == 8
    assert round_up(8) == 8
    assert round_up(9) == 16
    assert round_up(32) == 32
    assert round_up(33) == 40
    assert round_up(32.77) == 40
    assert round_up(31.999) == 32
    assert round_up(32 + 1e-12) == 32
    assert round_up(10, 4) == 12
    assert round_up(12, 4) == 12
    assert round_up(8.0) == 8 and type(round_up(8.0)) is int
    assert type(round_up(np.float64(9.5))) is int


def test_site_size_worked_examples():
    assert site_size(1074, 1) == (40, 32)
    assert site_size(600, 3) == (48, 16)
    assert site_size(64, 1) == (8, 8)
    assert site_size(1, 1) == (8, 8)
    assert site_size(20000, 1) == (96, 96)
    assert site_size(20000, 1, cap=64) == (64, 64)
    w, d = site_size(1074.0, 1.0)
    assert type(w) is int and type(d) is int


def test_site_size_invariants():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        area = rng.uniform(1, 12000)
        aspect = rng.uniform(1, 3)
        w, d = site_size(area, aspect)
        assert w % STEP == 0 and d % STEP == 0, (area, aspect, w, d)
        assert STEP <= w <= GRID_MAX and STEP <= d <= GRID_MAX, (area, aspect, w, d)
        if w < GRID_MAX and d < GRID_MAX:
            assert w * d >= area, (area, aspect, w, d)
            assert w * (d - STEP) < area, (area, aspect, w, d)
        if area <= GRID_MAX * GRID_MAX:
            assert w * d >= area, (area, aspect, w, d)


def test_all_alternates():
    alts = all_alternates()
    assert len(alts) == 72
    assert alts == sorted(alts)
    assert len(set(alts)) == 72
    assert all(recipe_by_id[a]["alternate"] for a in alts)
    assert "Recipe_Alternate_AdheredIronPlate_C" in alts
    assert all_alternates() == alts


def test_targets():
    ts = targets()
    assert len(ts) == 48
    assert ts == sorted(ts)
    assert targets() is ts
    for t in (RIP, "Desc_IronIngot_C", "Desc_Motor_C", "Desc_CircuitBoard_C", "Desc_Computer_C"):
        assert t in ts, t
    for t in ("Desc_AluminumIngot_C", "Desc_Fabric_C", "Desc_XmasBall1_C"):
        assert t not in ts, t
    assert not set(ts) & set(RESOURCE_LIMITS)


def test_random_alternates():
    alts = set(all_alternates())
    a = random_alternates(np.random.default_rng(7))
    b = random_alternates(np.random.default_rng(7))
    assert a == b
    assert type(a) is frozenset
    sizes = []
    for seed in range(300):
        s = random_alternates(np.random.default_rng(seed))
        assert type(s) is frozenset
        assert s <= alts
        sizes.append(len(s))
    assert min(sizes) <= 8, min(sizes)
    assert max(sizes) >= 64, max(sizes)
    assert 26 <= sum(sizes) / len(sizes) <= 46, sum(sizes) / len(sizes)
    assert len(set(sizes)) >= 30


def test_scale():
    m = {"a": 1.5, "b": 0.4}
    out = scale(m, 5)
    assert out == {"a": 7.5, "b": 2.0}
    assert m == {"a": 1.5, "b": 0.4}
    assert scale({}, 3) == {}
    assert scale(m, 1) == m and scale(m, 1) is not m


def test_machine_count():
    assert machine_count({"a": 7.5, "b": 2.0}) == 10
    assert machine_count({"a": 0.1}) == 1
    assert machine_count({"a": 2.0000000001}) == 2
    assert machine_count({"a": 1.9999}) == 2
    assert machine_count({}) == 0
    assert machine_count(scale({"a": 0.4}, 5)) == 2
    assert type(machine_count({"a": 2.5})) is int


def test_valid_rates_worked_examples():
    assert valid_rates({"Recipe_IngotIron_C": 1}) == [1, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 45]
    assert valid_rates({"Recipe_IngotIron_C": 0.4}) == RATES
    assert valid_rates({"Recipe_IngotIron_C": 1}, cap=40) == [1, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30]
    assert valid_rates({"Recipe_IngotIron_C": 100}) == []
    unit, _, _ = solve_lp("Desc_Motor_C", 1, set())
    assert valid_rates(unit) == [1, 2, 2.5, 3, 4, 5, 7.5]
    unit, _, _ = solve_lp("Desc_ModularFrameHeavy_C", 1, set())
    assert valid_rates(unit) == [1]


def test_valid_rates_invariants():
    for t in targets()[::3]:
        unit, _, _ = solve_lp(t, 1, None)
        rates = valid_rates(unit)
        assert rates == [r for r in RATES if r in rates], t
        for r in RATES:
            m = scale(unit, r)
            ok = machine_count(m) <= MAX_MACHINES and min_floor_area(m) <= GRID_MAX * GRID_MAX
            assert (r in rates) == ok, (t, r)


def test_min_sides():
    assert min_sides([]) == (0, 0)
    assert min_sides(["Recipe_IngotIron_C"]) == (24, 8)
    unit, _, _ = solve_lp(RIP, 1, set())
    assert min_sides(unit) == (32, 16)
    assert min_sides(list(unit)) == (32, 16)
    unit, _, _ = solve_lp("Desc_ModularFrameHeavy_C", 1, set())
    assert min_sides(unit) == (48, 24)
    for t in targets()[::4]:
        unit, _, _ = solve_lp(t, 1, None)
        long_side, short_side = min_sides(unit)
        assert long_side >= short_side and long_side % STEP == 0 and short_side % STEP == 0, t
        assert long_side <= GRID_MAX, t


def test_random_task_shape():
    for seed in range(60):
        task = random_task(np.random.default_rng(seed))
        assert set(task) == KEYS, task.keys()
        assert task["target"] in targets()
        assert task["rate"] in RATES
        assert task["floors"] == FLOORS
        assert type(task["width"]) is int and type(task["depth"]) is int and type(task["side"]) is int
        assert task["side"] in (0, 1, 2, 3)
        assert task["width"] % STEP == 0 and task["depth"] % STEP == 0
        assert STEP <= task["width"] <= GRID_MAX and STEP <= task["depth"] <= GRID_MAX
        assert type(task["alternates"]) is frozenset


def test_random_task_deterministic():
    a = [random_task(np.random.default_rng(s)) for s in range(10)]
    b = [random_task(np.random.default_rng(s)) for s in range(10)]
    assert a == b
    rng = np.random.default_rng(3)
    c = [random_task(rng) for _ in range(10)]
    assert len({(t["target"], t["rate"], t["width"], t["depth"], t["side"]) for t in c}) >= 8


def test_random_task_consistent():
    rng = np.random.default_rng(11)
    for _ in range(150):
        task = random_task(rng)
        machines, _, _ = solve_lp(task["target"], task["rate"], task["alternates"])
        assert machine_count(machines) <= MAX_MACHINES, task
        assert task["width"] * task["depth"] >= min_floor_area(machines) - 1e-6, task
        long_side, short_side = min_sides(machines)
        assert max(task["width"], task["depth"]) >= long_side, task
        assert min(task["width"], task["depth"]) >= short_side, task
        graph = build_graph(task["target"], task["rate"], alternates=task["alternates"])
        assert set(graph["nodes"]) == set(machines), task


def test_random_task_variety():
    rng = np.random.default_rng(5)
    tasks = [random_task(rng) for _ in range(300)]
    assert len({t["target"] for t in tasks}) >= 35
    assert {t["side"] for t in tasks} == {0, 1, 2, 3}
    assert any(t["width"] > t["depth"] for t in tasks)
    assert any(t["depth"] > t["width"] for t in tasks)
    assert any(t["width"] == GRID_MAX or t["depth"] == GRID_MAX for t in tasks)
    assert any(t["width"] * t["depth"] <= 400 for t in tasks)
    assert len({t["rate"] for t in tasks}) >= 12
    assert any(len(t["alternates"]) < 10 for t in tasks)
    assert any(len(t["alternates"]) > 60 for t in tasks)
    first, last = 0, 0
    for t in tasks:
        unit, _, _ = solve_lp(t["target"], 1, t["alternates"])
        rates = valid_rates(unit)
        first += t["rate"] == rates[0]
        last += len(rates) > 1 and t["rate"] == rates[-1]
    assert first >= 8 and last >= 8, (first, last)


def test_env_reset_never_fails_at_start():
    rng = np.random.default_rng(21)
    env = LayoutEnv()
    for _ in range(80):
        task = random_task(rng)
        env.reset(task)
        assert not env.done, task
        assert env.n_mask()[1], task


def test_random_agent_on_random_tasks():
    rng = np.random.default_rng(42)
    env = LayoutEnv()
    outcomes = {"ok": 0, "fail": 0}
    for _ in range(40):
        env.reset(random_task(rng))
        reward = 0.0
        while not env.done:
            ns = np.flatnonzero(env.n_mask())
            n = int(ns[rng.integers(len(ns))])
            spots = np.argwhere(env.pos_mask(n))
            rot, f, x, y = (int(v) for v in spots[rng.integers(len(spots))])
            reward = env.step(n, rot, f, x, y)
        outcomes["ok" if reward == 1.0 else "fail"] += 1
    assert outcomes["ok"] > 0 and outcomes["fail"] > 0, outcomes


if __name__ == "__main__":
    tests = [test_round_up, test_site_size_worked_examples, test_site_size_invariants,
             test_all_alternates, test_targets, test_random_alternates,
             test_scale, test_machine_count, test_valid_rates_worked_examples, test_valid_rates_invariants,
             test_min_sides, test_random_task_shape, test_random_task_deterministic,
             test_random_task_consistent, test_random_task_variety,
             test_env_reset_never_fails_at_start, test_random_agent_on_random_tasks]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")