import json

import pytest

from tools import registry
from tools.baselines import eval_rows, build, stage_of
from tools.digest import make_tasks, summarise, report, mismatch
from tools.replay import replay, broken
from tools.stats import mean, paired, verdict, band, bands

META = {"plans": "x.json", "k": 64, "seed": 0, "checkpoint": "last.pt", "note": "", "digested": "2026-01-01 00:00"}


@pytest.fixture(scope="module")
def rows():
    """A few stage-3 eval tasks: 4 A can solve (one with a 9+ block) and 1 it cannot"""
    every = eval_rows(3)
    solvable = [r for r in every if r["A"]["score"] > 0]
    big = [r for r in solvable if r["seed"] == 55040]
    return solvable[:3] + big + [r for r in every if r["A"]["score"] <= 0][:1]


@pytest.fixture(scope="module")
def base(rows):
    return build(3, rows)


def plan_file(rows, greedy="A", best="B"):
    """A best_of plan file that plays A's plans as greedy and B's as best"""
    return [{"seed": r["seed"], "greedy": {"plan": r[greedy]["plan"], "score": r[greedy]["score"]},
             "best": {"plan": r[best]["plan"], "score": r[best]["score"]}} for r in rows]


def test_verdict_rule():
    assert verdict(0.010, 0.004) == "helps"
    assert verdict(0.010, 0.006) == "inconclusive"
    assert verdict(0.004, 0.001) == "neutral"
    assert verdict(-0.004, 0.001) == "neutral"
    assert verdict(-0.010, 0.004) == "worse"
    assert verdict(-0.010, 0.006) == "inconclusive"
    assert verdict(None, None) is None


def test_paired_by_hand():
    p = paired([0.5, 0.6, 0.7], [0.4, 0.6, 0.4])
    assert p["n"] == 3 and p["better"] == 2 and p["worse"] == 0
    assert p["diff"] == pytest.approx(0.4 / 3)
    assert p["se"] == pytest.approx(0.0881917, abs=1e-6)
    assert p["verdict"] == "inconclusive"
    assert paired([], [])["diff"] is None
    assert paired([0.5], [0.4])["se"] is None


def test_bands():
    assert bands([5, 15]) == ["<5", "5-14", ">=15"]
    assert [band(r, [5, 15]) for r in (1, 4.9, 5, 14.5, 15, 30)] == ["<5", "<5", "5-14", "5-14", ">=15", ">=15"]
    assert mean([]) is None


def test_replay_reproduces_saved_scores(rows):
    for r in rows:
        for who in "AB":
            x = replay(r["task"], r[who]["plan"])
            assert x["score"] == pytest.approx(r[who]["score"], abs=1e-12)
            assert broken(x) == (r[who]["score"] <= 0)


def test_replay_valid_has_parts_and_broken_does_not_crash(rows):
    good = replay(rows[0]["task"], rows[0]["A"]["plan"])
    assert good["kind"] == "valid" and good["terms"]["length"] > 0
    assert good["blocks"] == [m[0] for m in rows[0]["A"]["plan"]]
    bad = replay(rows[0]["task"], rows[0]["A"]["plan"][:1])
    assert bad["kind"] == "error" and broken(bad)


def test_baselines_mark_solvable_and_big(base):
    assert [t["solvable"] for t in base["tasks"]] == [True, True, True, True, False]
    assert [t["big"] for t in base["tasks"]] == [False, False, False, True, False]


def test_stage_of(rows):
    every = eval_rows()
    assert stage_of([r["seed"] for r in rows], every) == 3
    with pytest.raises(ValueError):
        stage_of([50000, 55000], every)
    with pytest.raises(ValueError):
        stage_of([1], every)


def test_summary_of_a_run_that_plays_b(rows, base):
    tasks = make_tasks(plan_file(rows), rows)
    s = summarise("x", META, tasks, base, config=registry.load_config())
    assert s["tasks"] == 5 and s["solvable"] == 4
    assert s["greedy"]["mean"] == pytest.approx(mean(r["A"]["score"] for r in rows[:4]), abs=1e-5)
    assert s["best"]["mean"] == pytest.approx(mean(r["B"]["score"] for r in rows[:4]), abs=1e-5)
    assert s["vs"]["B"]["best"]["diff"] == 0 and s["vs"]["B"]["best"]["verdict"] == "neutral"
    assert s["vs"]["A"]["greedy"]["diff"] == 0
    assert s["best"]["broken"] == 0 and s["flags"] == [] and s["replay_gap"] == 0
    assert s["big_block"]["n"] == 1 and s["big_block"]["tasks"][0]["seed"] == 55040
    assert s["parts"]["best"] == s["parts"]["B"]
    assert s["unsolvable"]["n"] == 1
    assert "best-of-64" in report(s)


def test_broken_best_task_is_counted_and_flagged(rows, base):
    plans = plan_file(rows)
    plans[1]["best"] = plan_file(rows[4:], best="A")[0]["best"]
    tasks = make_tasks(plans, rows)
    tasks[1]["best"]["saved"] = tasks[1]["best"]["score"]
    s = summarise("x", META, tasks, base, config=registry.load_config())
    assert s["best"]["broken"] == 1
    assert [b["seed"] for b in s["broken"]["best"]] == [rows[1]["seed"]]
    assert any("broken in best-of-64" in x for x in s["flags"])
    assert s["vs"]["B"]["best"]["both_work"]["n"] == 3
    assert s["parts"]["best"]["valid"] == 3


def test_changed_scores_are_flagged(rows, base):
    plans = plan_file(rows)
    plans[0]["best"]["score"] += 0.01
    tasks = make_tasks(plans, rows)
    assert mismatch(tasks) == pytest.approx(0.01)
    s = summarise("x", META, tasks, base, config=registry.load_config())
    assert any("differ" in x for x in s["flags"])


def test_specialist_comparison(rows, base):
    spec = make_tasks(plan_file(rows, best="A"), rows)
    tasks = make_tasks(plan_file(rows), rows)
    cfg = registry.load_config()
    s = summarise("x", META, tasks, base, "s", spec, cfg)
    assert s["vs"]["specialist"]["best"]["diff"] == pytest.approx(s["vs"]["A"]["best"]["diff"])
    assert "specialist" in s["parts"]
    assert "not in the registry" in summarise("x", META, tasks, base, "s", None, cfg)["vs"]["specialist"]["note"]
    assert "is the specialist" in summarise("s", META, tasks, base, "s", None, cfg)["vs"]["specialist"]["note"]


def test_registry_round_trip(rows, base, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    tasks = make_tasks(plan_file(rows), rows)
    s = summarise("x", META, tasks, base, config=registry.load_config())
    registry.save_run(s, {"run": "x", "stage": 3, "source": META, "tasks": tasks})
    registry.save_run(s)
    assert registry.load_summary("x") == json.loads(json.dumps(s))
    assert registry.load_tasks("x")["tasks"][0]["best"]["plan"] == rows[0]["B"]["plan"]
    assert [r["run"] for r in registry.load_index()] == ["x"] and registry.all_runs() == ["x"]
    assert registry.load_summary("y") is None
    with pytest.raises(ValueError):
        registry.load_summary("../x")
