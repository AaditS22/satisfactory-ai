import datetime
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tools import registry, jobs, procs, rules, supervisor, digest
from tools.curve import read_rows

HERE = Path(__file__).parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "layout" / "train.py").exists())
SUP = registry.load_config()["supervisor"]


def fast_config():
    """The real config with short waits, so a whole fake night takes seconds"""
    cfg = registry.load_config()
    cfg["supervisor"].update(poll_s=0.3, min_free_ram_gb=0.05, kill_free_ram_gb=0.01, milestones=[20],
                             hang_s=3, start_grace_s=30, best_of_s=1)
    return cfg


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo: git, a fake layout.train / layout.best_of and a 6-task eval set"""
    (tmp_path / "layout").mkdir()
    (tmp_path / "layout" / "__init__.py").write_text("")
    shutil.copy(HERE / "fakes" / "fake_train.py", tmp_path / "layout" / "train.py")
    shutil.copy(HERE / "fakes" / "fake_best_of.py", tmp_path / "layout" / "best_of.py")
    rows = [r for r in json.loads((REPO / "data" / "eval_set.json").read_text()) if r["stage"] == 3]
    keep = [r for r in rows if r["A"]["score"] > 0][:5] + [r for r in rows if r["A"]["score"] <= 0][:1]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "eval_set.json").write_text(json.dumps(keep))
    (tmp_path / ".gitignore").write_text("runs/\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "start"]):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    procs.cleanup(tmp_path, everything=True)


def events(kind=None):
    files = sorted((jobs.DIR / "events").glob("*.json"))
    found = [json.loads(p.read_text()) for p in files]
    return [e for e in found if kind is None or e["kind"] == kind]


def queue(run, note, rounds=60, prereg=""):
    return jobs.add(jobs.new_entry(run, 3, {"rounds": rounds, "workers": 2}, note, prereg, sup=SUP), SUP)


def test_classify_and_declared_parent():
    win = ["C:\\repo\\.venv\\Scripts\\python.exe", "-c",
           "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=4321, pipe_handle=880)",
           "--multiprocessing-fork"]
    assert procs.classify(win) == "worker" and procs.declared_parent(win) == 4321
    assert procs.classify(["python", "-m", "layout.train", "--run", "x"]) == "train"
    assert procs.classify(["python.exe", "C:\\repo\\layout\\best_of.py", "last.pt"]) == "best_of"
    assert procs.classify(["python", "-m", "tensorboard.main"]) is None
    assert procs.declared_parent(["python", "-m", "layout.train"]) is None


def test_queue_settings_cover_every_train_flag():
    source = (REPO / "layout" / "train.py").read_text()
    flags = set(re.findall(r'add_argument\("(--[a-zA-Z-]+)"', source))
    known = {jobs.flag(k) for k in jobs.SETTINGS} | {"--run", "--stage", "--resume", "--note"}
    assert flags == known


def test_new_entry_rules(repo):
    ok = jobs.new_entry("r1", 3, {"rounds": 2000, "workers": 16, "solvable": True, "cost_map": False, "copies": 8,
                                  "fixed_scale": True, "scale": 0.061034, "seed": 1, "ent_coef": 0.01}, "a note", sup=SUP)
    cmd = jobs.command(ok)
    assert cmd[1:7] == ["-m", "layout.train", "--run", "r1", "--stage", "3"]
    assert "--solvable" in cmd and "--cost-map" not in cmd and cmd[-2:] == ["--note", "a note"]
    assert cmd[cmd.index("--scale") + 1] == "0.061034" and cmd[cmd.index("--ent-coef") + 1] == "0.01"
    bad = [dict(run="../x"), dict(note=" "), dict(stage=9), dict(settings={"rounds": 99999, "workers": 16}),
           dict(settings={"workers": 24}), dict(settings={"workers": 16, "scale": 0.1}),
           dict(settings={"workers": 16, "fixed_scale": True}), dict(settings={"workers": 16, "bogus": 1}),
           dict(settings={"workers": 16, "rounds": "100"}), dict(settings={"workers": 16, "init": "../../x.pt"}),
           dict(settings={"workers": 16, "init": "missing.pt"}), dict(settings={})]
    for change in bad:
        args = {"run": "r1", "stage": 3, "settings": {"workers": 16}, "note": "n", **change}
        with pytest.raises(ValueError):
            jobs.new_entry(args["run"], args["stage"], args["settings"], args["note"], sup=SUP)
    (repo / "runs" / "other").mkdir(parents=True)
    (repo / "runs" / "other" / "w.pt").write_text("x")
    assert jobs.new_entry("r1", 3, {"workers": 16, "init": "runs/other/w.pt"}, "n", sup=SUP)["settings"]["init"] \
        == "runs/other/w.pt"


def test_queue_file(repo):
    a = queue("r1", "one")
    with pytest.raises(ValueError):
        queue("r1", "again")
    b = queue("r2", "two")
    assert [e["id"] for e in jobs.load()["entries"]] == [a["id"], b["id"]] == [1, 2]
    assert jobs.remove(1)["run"] == "r1" and jobs.remove(1) is None
    assert queue("r3", "three")["id"] == 3
    small = {**SUP, "max_queue": 2}
    with pytest.raises(ValueError):
        jobs.add(jobs.new_entry("r4", 3, {"workers": 2}, "n", sup=small), small)


def test_resume_uses_the_runs_own_settings(repo):
    folder = repo / "runs" / "stage3" / "old"
    folder.mkdir(parents=True)
    (folder / "last.pt").write_text("x")
    (folder / "run.json").write_text(json.dumps({"config": {
        "stage": 3, "rounds": 2000, "n_games": 48, "workers": 16, "T": 32, "lr": 2.5e-4, "epochs": 4, "minibatch": 256,
        "ent_coef": 0.01, "seed": 1, "init": "runs/other/f.pt", "solvable": True, "overlap": False, "shaping": 0.0,
        "eval_every": 10, "images_every": 0, "copies": 8, "groups": 14, "broken_margin": 0.1, "fixed_scale": True,
        "set_scale": 0.061034, "cost_map": True}}))
    cmd = jobs.command(jobs.resume_entry("old", sup=SUP))
    assert cmd[-1] == "--resume" and cmd[cmd.index("--seed") + 1] == "1" and "--cost-map" in cmd
    assert cmd[cmd.index("--games") + 1] == "48" and "--fixed-scale" in cmd
    assert not {"--init", "--note", "--scale", "--overlap"} & set(cmd)
    with pytest.raises(ValueError):
        jobs.resume_entry("nope", sup=SUP)
    with pytest.raises(ValueError):
        jobs.new_entry("old", 3, {"workers": 16}, "n", sup=SUP)


def test_rules():
    flags = registry.load_config()["flags"]
    row = lambda r, **k: {"round": r, "score": 0.5, "entropy": 1.3, "valid_score": 0.6, "broken": 0.06,
                          "approx_kl": 0.05, "value_loss": 0.1, "policy_loss": 0.0, **k}
    good = [row(r) for r in range(100)]
    assert rules.check(good, [], flags, SUP) == [] and rules.check([], [], flags, SUP) == []
    many_broken = good[:80] + [row(r, broken=0.4) for r in range(80, 100)]
    assert [x[0] for x in rules.check(many_broken, [], flags, SUP)] == ["broken"]
    collapsed = good[:90] + [row(r, entropy=0.1) for r in range(90, 100)]
    assert [x[0] for x in rules.check(collapsed, [], flags, SUP)] == ["entropy_collapse"]
    assert rules.check([row(r, entropy=0.1) for r in range(20)], [], flags, SUP) == []
    drifting = good[:5] + [row(r, entropy=1.8, valid_score=0.55) for r in range(5, 40)]
    assert [x[0] for x in rules.check(drifting, [], flags, SUP)] == ["drift"]
    assert rules.check(good[:50] + [row(50, entropy=float("nan"))], [], flags, SUP)[0][0] == "nan"
    waits = {"hang_s": 100, "start_grace_s": 300}
    assert rules.hang(101, True, waits)[0] == "hang" and rules.hang(101, False, waits) is None
    assert rules.hang(301, False, waits)[0] == "hang" and rules.hang(5, True, waits) is None
    assert rules.low_ram(0.5, SUP)[0] == "low_ram" and rules.low_ram(8, SUP) is None


def test_deadline_text():
    now = datetime.datetime(2026, 11, 16, 23, 30)
    assert supervisor.parse_deadline("09:00", now) == datetime.datetime(2026, 11, 17, 9, 0)
    assert supervisor.parse_deadline("23:45", now) == datetime.datetime(2026, 11, 16, 23, 45)
    assert supervisor.parse_deadline("2026-11-18 07:15", now) == datetime.datetime(2026, 11, 18, 7, 15)
    assert supervisor.parse_deadline(None) is None


def test_a_whole_night(repo):
    """Good run, drift kill, hang kill, a crash that leaves workers behind, then a clean last run"""
    (repo / "new_idea.py").write_text("print('not committed')\n")
    queue("good", "plain run", prereg="good should finish")
    queue("drifty", "drift please", rounds=2000)
    queue("hangs", "hang please", rounds=2000)
    queue("leaver", "orphan please", rounds=2000)
    queue("after", "plain run again")
    git_index = (repo / ".git" / "index").read_bytes()
    supervisor.supervise(fast_config())

    kinds = [(e["kind"], e["run"]) for e in events()]
    assert ("finished", "good") in kinds and ("finished", "after") in kinds
    assert ("milestone", "good") in kinds and kinds[-1] == ("queue_empty", None)
    assert {(e["run"], e["rule"]) for e in events("killed")} == {("drifty", "drift"), ("hangs", "hang")}
    assert [e["run"] for e in events("crashed")] == ["leaver"]
    assert "leftover worker" in (jobs.DIR / "supervisor.log").read_text()
    assert procs.ours(repo) == [] and jobs.load()["entries"] == []

    good = registry.load_summary("good")
    assert good["status"] == "finished" and good["solvable"] == 5 and good["best"]["broken"] == 0
    assert good["training"]["complete"] and good["prereg"]["text"] == "good should finish"
    assert events("finished")[0]["best"] == good["best"]["mean"]
    patch = registry.run_file("good", "code.patch").read_text()
    assert "new_idea.py" in patch and "not committed" in patch
    assert (repo / ".git" / "index").read_bytes() == git_index
    assert not (jobs.DIR / "plans" / "good.json").exists()

    drifty = registry.load_summary("drifty")
    assert drifty["status"] == "killed" and drifty["reason"].startswith("drift:") and "best" not in drifty
    assert drifty["training"]["rounds_done"] < 200
    assert registry.load_summary("leaver")["status"] == "crashed"
    rows = {r["run"]: r for r in registry.load_index()}
    assert rows["good"]["best"] is not None and rows["hangs"]["best"] is None and rows["hangs"]["status"] == "killed"
    assert "KILLED" in digest.report(drifty)
    milestone = events("milestone")[0]
    assert milestone["milestone"] == 20 and milestone["round"] >= 20 and milestone["rounds"] == 60


def test_deadline_skips_runs_that_do_not_fit(repo):
    queue("long", "plain", rounds=5000)
    queue("short", "plain", rounds=30)
    cfg = fast_config()
    supervisor.supervise(cfg, datetime.datetime.now() + datetime.timedelta(minutes=5))
    kinds = [(e["kind"], e["run"]) for e in events()]
    assert ("finished", "short") in kinds and ("started", "long") not in kinds
    assert kinds[-2:] == [("deadline", None), ("queue_empty", None)]
    assert [e["run"] for e in jobs.load()["entries"]] == ["long"]


def test_stop_request_and_resume(repo):
    queue("stopme", "plain", rounds=2000)
    queue("next", "plain", rounds=30)

    def ask():
        time.sleep(2.0)
        registry.write_json(jobs.DIR / "stop.json", {"what": "run", "reason": "looks bad"})
    threading.Thread(target=ask).start()
    supervisor.supervise(fast_config())
    killed = events("killed")[0]
    assert killed["run"] == "stopme" and killed["rule"] == "stop" and killed["why"] == "looks bad"
    assert events("finished")[0]["run"] == "next" and procs.ours(repo) == []

    queue("half", "stop_half please", rounds=40)
    supervisor.supervise(fast_config())
    assert events("crashed")[-1]["run"] == "half" and registry.load_summary("half")["status"] == "crashed"
    jobs.add(jobs.resume_entry("half", sup=SUP), SUP)
    supervisor.supervise(fast_config())
    assert events("finished")[-1]["run"] == "half"
    assert [r["round"] for r in read_rows(repo / "runs" / "stage3" / "half" / "log.csv")] == list(range(40))
    assert registry.load_summary("half")["status"] == "finished"


def test_refuses_when_ram_is_low_or_a_training_is_running(repo):
    queue("r1", "plain", rounds=30)
    cfg = fast_config()
    cfg["supervisor"]["min_free_ram_gb"] = 100000
    supervisor.supervise(cfg)
    assert "RAM" in events("refused")[0]["why"] and [e["run"] for e in jobs.load()["entries"]] == ["r1"]
    other = subprocess.Popen([procs.psutil.Process().exe(), "-m", "layout.train", "--run", "manual", "--stage", "3",
                              "--rounds", "5000", "--note", "by hand"])
    time.sleep(1.5)
    supervisor.supervise(fast_config())
    assert "another training" in events("refused")[-1]["why"] and other.poll() is None
    assert [e["run"] for e in jobs.load()["entries"]] == ["r1"]
    procs.kill_tree(other.pid)
