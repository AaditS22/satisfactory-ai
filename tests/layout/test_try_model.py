import tempfile
from pathlib import Path

import torch

from layout.api import evaluate
from layout.evaluate import load_eval
from layout.policy import Policy
from layout.try_model import try_model, task_name, list_tasks
from layout.view import draw_compare


def small_row():
    """The stage 1 test task with the fewest machines (quick to play)"""
    return min(load_eval(1)[:10], key=lambda r: len(r["A"]["plan"]))


def test_try_model_plays_greedy_and_keeps_best_sample():
    torch.manual_seed(0)
    policy = Policy().eval()
    row = small_row()
    g, best = try_model(policy, row, k=3, seed=0, workers=1, slots=2)
    for info in (g, best):
        assert "plan" in info and "score" in info
        assert abs(evaluate(row["task"], info["plan"])["score"] - info["score"]) < 1e-9


def test_try_model_same_seed_same_best():
    torch.manual_seed(0)
    policy = Policy().eval()
    row = small_row()
    a = try_model(policy, row, k=3, seed=5, workers=1, slots=2)[1]
    b = try_model(policy, row, k=3, seed=5, workers=1, slots=2)[1]
    assert a["plan"] == b["plan"]


def test_task_name_and_list(capsys):
    row = small_row()
    assert f"{row['task']['width']} x {row['task']['depth']} m site" in task_name(row)
    list_tasks([row])
    assert str(row["seed"]) in capsys.readouterr().out


def test_draw_compare_writes_png():
    row = small_row()
    a, b = evaluate(row["task"], row["A"]["plan"]), evaluate(row["task"], row["B"]["plan"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cmp.png"
        draw_compare([("A", a, False), ("B", b, True)], path, task_name(row))
        assert path.stat().st_size > 10_000
