import math

import numpy as np

from layout.runlog import PARTS, mean, kind, round_stats, RunLog, read_log, line, tag


def parts(v):
    return {p: v for p in PARTS}


def valid(score, v, blocks=3):
    return {"score": score, "failed": False, "blocks": blocks,
            "terms": {"passed": True, "routed": 1.0, "fraction": 1.0, **parts(v)}}


def invalid(score, routed, blocks=4):
    return {"score": score, "failed": False, "blocks": blocks,
            "terms": {"passed": False, "routed": routed, "fraction": 0.5, **parts(9.0)}}


def failed(score, blocks=2):
    return {"score": score, "failed": True, "blocks": blocks, "terms": {"placed": 0.5}}


def test_mean():
    assert mean([1, 2, 3]) == 2.0 and isinstance(mean([1]), float)
    assert math.isnan(mean([]))


def test_kind():
    assert kind(valid(0.6, 1.0)) == "valid"
    assert kind(invalid(-0.5, 0.5)) == "invalid"
    assert kind(failed(-1.5)) == "failed"


def test_round_stats_hand():
    s = round_stats([valid(0.6, 1.0, 3), valid(0.8, 0.0, 5), invalid(-0.5, 0.5, 4), failed(-1.5, 2)])
    assert s["games"] == 4
    assert abs(s["score"] - (-0.15)) < 1e-9
    assert s["failed"] == 0.25 and s["invalid"] == 0.25
    assert abs(s["valid_score"] - 0.7) < 1e-9
    assert s["blocks"] == 3.5
    assert all(s[p] == 0.5 for p in PARTS)
    assert abs(s["routed"] - 2.5 / 3) < 1e-9 and abs(s["fraction"] - 2.5 / 3) < 1e-9


def test_round_stats_all_floats():
    s = round_stats([valid(0.6, 1.0)])
    assert all(isinstance(v, float) for k, v in s.items() if k != "games") and isinstance(s["games"], int)


def test_round_stats_no_valid_games():
    s = round_stats([failed(-1.2), invalid(-0.4, 0.2)])
    assert math.isnan(s["valid_score"]) and all(math.isnan(s[p]) for p in PARTS)
    assert s["failed"] == 0.5 and s["routed"] == 0.2


def test_round_stats_empty():
    s = round_stats([])
    assert s["games"] == 0 and math.isnan(s["score"]) and math.isnan(s["failed"])


def test_runlog_roundtrip(tmp="tests/_tmp_runlog/log.csv"):
    import shutil
    shutil.rmtree("tests/_tmp_runlog", ignore_errors=True)
    log = RunLog(tmp, tensorboard=False)
    log.write({"round": 0, "score": 0.512345678, "kl": float("nan")})
    log.write({"round": 1, "score": 0.6, "kl": 0.002})
    back = read_log(tmp)
    assert list(back) == ["round", "score", "kl"]
    assert np.array_equal(back["round"], [0, 1])
    assert abs(back["score"][0] - 0.51235) < 1e-9 and math.isnan(back["kl"][0]) and back["kl"][1] == 0.002
    RunLog(tmp, tensorboard=False).write({"round": 0, "score": 0.1, "kl": 0.0})
    assert len(read_log(tmp)["round"]) == 1
    shutil.rmtree("tests/_tmp_runlog")


def test_runlog_append_keeps_old_rows(tmp="tests/_tmp_runlog2/log.csv"):
    import shutil
    shutil.rmtree("tests/_tmp_runlog2", ignore_errors=True)
    RunLog(tmp, tensorboard=False).write({"round": 0, "score": 0.5})
    RunLog(tmp, tensorboard=False, append=True).write({"round": 1, "score": 0.6})
    back = read_log(tmp)
    assert np.array_equal(back["round"], [0, 1]) and np.allclose(back["score"], [0.5, 0.6])
    RunLog("tests/_tmp_runlog2/new.csv", tensorboard=False, append=True).write({"round": 0, "score": 0.1})
    assert np.array_equal(read_log("tests/_tmp_runlog2/new.csv")["round"], [0])
    shutil.rmtree("tests/_tmp_runlog2")


def test_tag_groups():
    assert tag("score") == "score/score" and tag("valid_score") == "score/valid_score"
    assert tag("invalid") == "rates/invalid" and tag("p_floors") == "penalty/p_floors"
    assert tag("entropy") == "ppo/entropy" and tag("approx_kl") == "ppo/approx_kl"
    assert tag("update_s") == "time/update_s" and tag("blocks") == "games/blocks"


def test_line():
    row = {"round": 3, **round_stats([valid(0.6, 1.0)]), "entropy": 2.5, "approx_kl": 0.003, "clip_frac": 0.05,
           "value_loss": 0.1, "collect_s": 1.2, "update_s": 3.1}
    text = line(row)
    assert text.startswith("round    3") and "score 0.600" in text and "learn 3.1 s" in text


TESTS = [test_mean, test_kind, test_round_stats_hand, test_round_stats_all_floats, test_round_stats_no_valid_games,
    test_round_stats_empty, test_runlog_roundtrip, test_runlog_append_keeps_old_rows,
    test_tag_groups,
    test_line]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")