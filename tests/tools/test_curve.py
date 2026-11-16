import csv
import json

import pytest

from tools import registry
from tools.curve import read_rows, avg, blocks, drift, flags, training

CFG = {"drift_entropy": 0.3, "drift_valid": 0.02, "entropy_collapse": 0.3, "broken_rise": 0.02}


def row(r, entropy=1.3, valid=0.6, broken=0.06, score=0.5):
    return {"round": r, "score": score, "failed": broken / 2, "invalid": broken / 2, "valid_score": valid,
            "entropy": entropy, "approx_kl": 0.05, "value_loss": 0.1, "policy_loss": 0.01, "collect_s": 3.0,
            "update_s": 3.0, "push": 0.7, "flat": 0.01, "clip_frac": 0.2, "scale": 0.061}


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_run(folder, rows, evals, rounds=200):
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "log.csv", rows)
    write(folder / "eval.csv", evals)
    (folder / "run.json").write_text(json.dumps({"note": "n", "started": "s", "command": "c", "code": "abc",
                                                 "config": {"rounds": rounds}, "resumed": []}))


def good(n=200):
    return [row(r, entropy=1.3 - 0.7 * r / n, valid=0.59 + 0.02 * r / n, broken=0.07 - 0.02 * r / n) for r in range(n)]


def evals(n=200, start=0.5, slope=0.0):
    return [{"round": r, "solv_score": start + slope * r, "gate_A": 0.4, "failed": 0.0, "invalid": 0.05}
            for r in range(0, n, 10)]


def loaded(tmp_path, rows, ev):
    write(tmp_path / "log.csv", rows)
    write(tmp_path / "eval.csv", ev)
    return read_rows(tmp_path / "log.csv"), read_rows(tmp_path / "eval.csv")


def test_read_rows_keeps_the_later_row_after_a_resume(tmp_path):
    rows = [row(0), row(1, entropy=9.0), row(2), row(1, entropy=1.0), row(2, entropy=2.0)]
    write(tmp_path / "log.csv", rows)
    got = read_rows(tmp_path / "log.csv")
    assert [r["round"] for r in got] == [0, 1, 2]
    assert got[1]["entropy"] == 1.0 and got[2]["entropy"] == 2.0
    assert got[0]["broken"] == pytest.approx(0.06) and got[0]["round_s"] == 6.0
    assert read_rows(tmp_path / "missing.csv") == []


def test_avg_and_blocks(tmp_path):
    rows, _ = loaded(tmp_path, [row(r, entropy=float(r)) for r in range(20)], evals(20))
    assert avg(rows, "entropy", 0, 4) == 2.0
    assert avg(rows, "entropy", 100, 200) is None
    b = blocks(rows, ["entropy"], 20, 2)
    assert [x["rounds"] for x in b] == ["0-9", "10-19"] and b[1]["entropy"] == 14.5


def test_good_run_has_no_flags(tmp_path):
    rows, ev = loaded(tmp_path, good(), evals())
    assert flags(rows, ev, 200, CFG) == []


def test_drift_needs_all_three_signs(tmp_path):
    drifting = [row(r) for r in range(5)] + [row(r, entropy=1.7, valid=0.57) for r in range(5, 60)]
    rows, ev = loaded(tmp_path, drifting, evals(60))
    assert drift(rows, ev, CFG)["rounds"] == "10-19"
    assert any("drift" in x for x in flags(rows, ev, 60, CFG))
    rows, ev = loaded(tmp_path, drifting, evals(60, slope=0.002))
    assert drift(rows, ev, CFG) is None
    only_entropy = [row(r) for r in range(5)] + [row(r, entropy=1.7) for r in range(5, 60)]
    rows, ev = loaded(tmp_path, only_entropy, evals(60))
    assert drift(rows, ev, CFG) is None
    small = [row(r) for r in range(5)] + [row(r, entropy=1.55, valid=0.57) for r in range(5, 60)]
    rows, ev = loaded(tmp_path, small, evals(60))
    assert drift(rows, ev, CFG) is None


def test_other_flags(tmp_path):
    rows, ev = loaded(tmp_path, good(120), evals(120))
    assert any("incomplete run: stopped at round 119 of 200" in x for x in flags(rows, ev, 200, CFG))
    collapsed = [row(r, entropy=1.3 * (1 - r / 200) ** 3) for r in range(200)]
    rows, ev = loaded(tmp_path, collapsed, evals())
    assert any("entropy collapse" in x for x in flags(rows, ev, 200, CFG))
    rising = [row(r, broken=0.05 + 0.05 * r / 200) for r in range(200)]
    rows, ev = loaded(tmp_path, rising, evals())
    assert any("rising broken share" in x for x in flags(rows, ev, 200, CFG))
    nan = good()
    nan[50]["entropy"] = float("nan")
    rows, ev = loaded(tmp_path, nan, evals())
    assert any("entropy is nan or infinite at round 50" in x for x in flags(rows, ev, 200, CFG))
    assert flags([], [], 200, CFG) == ["no training rows found"]


def test_training_record(tmp_path):
    make_run(tmp_path / "r", good(), evals(slope=0.0005))
    t = training(tmp_path / "r", registry.load_config())
    assert t["rounds_done"] == 200 and t["complete"] and t["flags"] == []
    assert t["scale"] == 0.061 and t["note"] == "n"
    assert len(t["train_blocks"]) == 10 and t["train_blocks"][0]["rounds"] == "0-19"
    assert t["eval"]["first"] == 0.5 and t["eval"]["best"]["round"] == 190
    assert t["entropy"]["rounds_0_4"] > t["entropy"]["last_10"]
    assert training(tmp_path / "nothing", registry.load_config()) is None
