import pytest

from tools import registry
from tools.baselines import eval_rows, build, fingerprint
from tools.compare import compare_runs, compare_set, task_view, runs_text, set_text, task_text, list_text
from tools.digest import make_tasks, resummarise

META = {"plans": "x.json", "k": 64, "seed": 0, "checkpoint": "last.pt", "note": "", "digested": "2026-01-01 00:00"}


def plan_file(rows, greedy, best):
    return [{"seed": r["seed"], "greedy": {"plan": r[greedy]["plan"], "score": r[greedy]["score"]},
             "best": {"plan": r[best]["plan"], "score": r[best]["score"]}} for r in rows]


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    """A small registry: run 'a' plays A's plans, run 'b' plays B's, run 'mix' plays B's with one task broken"""
    root = tmp_path_factory.mktemp("registry")
    mp = pytest.MonkeyPatch()
    mp.setattr(registry, "ROOT", root)
    mp.setattr(registry, "specialists", lambda: {3: "a"})
    every = eval_rows(3)
    solvable = [r for r in every if r["A"]["score"] > 0]
    rows = solvable[:4] + [r for r in solvable if r["seed"] == 55040] + [r for r in every if r["A"]["score"] <= 0][:1]
    base = build(3, rows)
    registry.write_json(root / "baselines" / "stage3.json", base)
    assert base["eval_set"] == fingerprint()
    files = {"a": plan_file(rows, "A", "A"), "b": plan_file(rows, "A", "B"), "mix": plan_file(rows, "A", "B")}
    files["mix"][1]["best"] = {"plan": rows[5]["A"]["plan"], "score": rows[5]["A"]["score"]}
    for run, plans in files.items():
        tasks = make_tasks(plans, rows)
        for t in tasks:
            t["best"]["saved"] = t["best"]["score"]
        stored = {"run": run, "stage": 3, "source": META, "tasks": tasks, "training": None}
        registry.save_run(resummarise(stored, base), stored)
    yield rows
    mp.undo()


def test_compare_two_runs(reg):
    c = compare_runs("b", "a")
    want = sum(r["B"]["score"] - r["A"]["score"] for r in reg[:5]) / 5
    assert c["all"]["n"] == 5 and c["all"]["diff"] == pytest.approx(want, abs=1e-5)
    assert c["big_block"]["n"] == 1 and c["big_block"]["tasks"][0]["seed"] == 55040
    assert sum(g["n"] for g in c["by_rate"].values()) == 5
    assert compare_runs("a", "b")["all"]["diff"] == pytest.approx(-want, abs=1e-5)
    assert "a = b, b = a" in runs_text(c)


def test_compare_with_baselines_and_specialist(reg):
    assert compare_runs("b", "B")["all"]["diff"] == 0
    assert compare_runs("a", "A")["all"]["verdict"] == "neutral"
    assert compare_runs("b", "specialist")["all"]["diff"] == compare_runs("b", "a")["all"]["diff"]
    assert compare_runs("b", "B", key="greedy")["all"]["diff"] == compare_runs("a", "B")["all"]["diff"]


def test_broken_task_shows_up(reg):
    c = compare_runs("mix", "b")
    assert c["broken"] == {"a": [reg[1]["seed"]], "b": []}
    assert c["all"]["a_broken"] == 1 and c["all"]["both_work"]["n"] == 4 and c["all"]["both_work"]["diff"] == 0
    assert c["a_worst"][0]["seed"] == reg[1]["seed"] and c["a_best"] == []


def test_set_takes_the_best_real_score_per_task(reg):
    s = compare_set(["mix", "a"], "B")
    best = [max(r["A"]["score"], r["B"]["score"]) if i != 1 else r["A"]["score"] for i, r in enumerate(reg[:5])]
    assert s["set"]["a"] == pytest.approx(sum(best) / 5, abs=1e-5)
    assert s["set"]["a_broken"] == 0 and sum(s["wins"].values()) == 5 and s["wins"]["a"] >= 1
    assert compare_set(["b"], "B")["set"]["diff"] == 0
    assert "holds the best score" in set_text(s)


def test_task_view(reg):
    v = task_view(55040)
    assert v["big_block"] and set(v["runs"]) == {"a", "b", "mix"}
    assert v["runs"]["b"]["best"]["plan"] == reg[4]["B"]["plan"]
    assert "plan" not in task_view(55040, ["a"], plans=False)["A"]
    assert "55040" in task_text(v) and "mix" in list_text()


def test_errors(reg):
    with pytest.raises(ValueError):
        compare_runs("b", "nope")
    with pytest.raises(ValueError):
        compare_runs("A", "B")
    with pytest.raises(ValueError):
        task_view(5)


def test_server_tools_answer_with_json_text(reg):
    pytest.importorskip("mcp")
    import json
    from tools import server
    runs = json.loads(server.list_runs())
    assert [r["run"] for r in runs["runs"]] == ["a", "b", "mix"] and runs["specialists"] == {"3": "a"}
    assert json.loads(server.list_runs(stage=1))["runs"] == []
    assert json.loads(server.get_run("b"))["best"]["broken"] == 0
    assert len(json.loads(server.get_run("b", per_task=True))["per_task"]) == 6
    assert json.loads(server.compare_runs("b", "B"))["all"]["diff"] == 0
    assert json.loads(server.compare_set(["a", "b"]))["set"]["n"] == 5
    assert set(json.loads(server.get_task(55040))["runs"]) == {"a", "b", "mix"}
    assert "tie" in json.loads(server.get_config())["config"]
    assert "not in the registry" in json.loads(server.get_run("nope"))["error"]
    assert "layouts" in json.loads(server.compare_runs("a", "b", "x"))["error"]
    assert "bad run name" in json.loads(server.get_run("../x"))["error"]
