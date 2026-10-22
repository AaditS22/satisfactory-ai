import shutil
from pathlib import Path

import numpy as np

from layout.evaluate import load_eval, play_eval, eval_stats, best_order, save_best, evaluate


def first_legal(policy, obs_list, masks):
    assert len(obs_list) == len(masks) and masks.any(1).all()
    return np.array([int(np.flatnonzero(m)[0]) for m in masks])


def result(score, k, a, b=0.0):
    return {"score": score, "kind": k, "A": a, "B": b, "gym": None}


def test_load_eval_stage_one():
    rows = load_eval(1)
    assert len(rows) == 30 and all(r["stage"] == 1 for r in rows)
    assert all(r["task"]["alternates"] is None or isinstance(r["task"]["alternates"], frozenset) for r in rows)
    assert all(isinstance(r["A"]["score"], float) and isinstance(r["B"]["score"], float) for r in rows)


def test_load_eval_other_stage():
    rows = load_eval(2)
    assert len(rows) == 20 and {r["target"] for r in rows} == {"Desc_IronPlateReinforced_C", "Desc_Rotor_C"}


def test_play_eval_results():
    rows = load_eval(1)[:4]
    res = play_eval(None, rows, first_legal)
    assert len(res) == 4
    for x, r in zip(res, rows):
        assert x["kind"] in ("valid", "invalid", "failed") and x["A"] == r["A"]["score"] and x["B"] == r["B"]["score"]
        assert x["gym"].env.done and x["gym"].task["target"] == r["target"]
        assert (x["score"] > 0) == (x["kind"] == "valid")


def test_play_eval_repeatable():
    rows = load_eval(1)[4:7]
    a = [x["score"] for x in play_eval(None, rows, first_legal)]
    b = [x["score"] for x in play_eval(None, rows, first_legal)]
    assert a == b


def test_play_eval_passes_policy_and_shrinks():
    sizes = []

    def spy(policy, obs_list, masks):
        assert policy == "net"
        sizes.append(len(obs_list))
        return first_legal(policy, obs_list, masks)

    play_eval("net", load_eval(1)[:3], spy)
    assert sizes[0] == 3 and sizes[-1] >= 1 and sizes == sorted(sizes, reverse=True)


def test_eval_stats_hand():
    s = eval_stats([result(0.7, "valid", 0.6, 0.8), result(0.5, "valid", 0.6, 0.6), result(-0.4, "invalid", 0.2),
                    result(-1.5, "failed", -1.6)])
    assert abs(s["score"] - (-0.175)) < 1e-9 and abs(s["A"] - (-0.05)) < 1e-9 and abs(s["B"] - 0.35) < 1e-9
    assert s["beats_A"] == 0.5 and abs(s["valid_score"] - 0.6) < 1e-9
    assert s["failed"] == 0.25 and s["invalid"] == 0.25


def test_eval_stats_tie_is_not_a_win():
    s = eval_stats([result(0.6, "valid", 0.6), result(0.61, "valid", 0.6)])
    assert s["beats_A"] == 0.5


def test_best_order():
    res = [result(0.2, "valid", 0), result(-1.0, "failed", 0), result(0.7, "valid", 0), result(0.5, "valid", 0)]
    assert best_order(res, 2) == [2, 3]
    assert best_order(res, 10) == [2, 3, 0, 1]


def test_save_best_and_evaluate():
    folder = Path("tests/_tmp_eval")
    shutil.rmtree(folder, ignore_errors=True)
    rows = load_eval(1)[:3]
    stats, paths = evaluate(None, rows, folder / "r0", first_legal, k=2)
    assert set(stats) == {"score", "A", "B", "beats_A", "valid_score", "failed", "invalid"}
    assert [p.name for p in paths] == ["best_0.png", "best_1.png"] and all(p.exists() for p in paths)
    res = play_eval(None, rows, first_legal)
    paths = save_best(res, folder / "r1", k=10)
    assert len(paths) == 3
    stats, paths = evaluate(None, rows, folder / "r2", first_legal, k=0)
    assert paths == [] and not (folder / "r2").exists() and "score" in stats
    shutil.rmtree(folder)


TESTS = [test_load_eval_stage_one, test_load_eval_other_stage, test_play_eval_results, test_play_eval_repeatable,
    test_play_eval_passes_policy_and_shrinks, test_eval_stats_hand,
    test_eval_stats_tie_is_not_a_win, test_best_order, test_save_best_and_evaluate]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")