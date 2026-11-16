import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import registry, jobs, procs, control, supervisor

HERE = Path(__file__).parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "layout" / "train.py").exists())


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo with the real layout code but a fake layout.train / layout.best_of, and fast settings"""
    shutil.copytree(REPO / "layout", tmp_path / "layout", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(HERE / "fakes" / "fake_train.py", tmp_path / "layout" / "train.py")
    shutil.copy(HERE / "fakes" / "fake_best_of.py", tmp_path / "layout" / "best_of.py")
    (tmp_path / "data").mkdir()
    for f in (REPO / "data").glob("*.json"):
        if f.name != "eval_set.json" and f.stat().st_size < 5_000_000:
            shutil.copy(f, tmp_path / "data" / f.name)
    rows = [r for r in json.loads((REPO / "data" / "eval_set.json").read_text()) if r["stage"] == 3]
    keep = [r for r in rows if r["A"]["score"] > 0][:5] + [r for r in rows if r["A"]["score"] <= 0][:1]
    (tmp_path / "data" / "eval_set.json").write_text(json.dumps(keep))
    cfg = registry.load_config()
    cfg["supervisor"].update(poll_s=0.3, min_free_ram_gb=0.05, kill_free_ram_gb=0.01, milestones=[20], best_of_s=1)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("TOOLS_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    state = registry.read_json(jobs.DIR / "state.json", {})
    if control.supervisor_alive():
        procs.kill_tree(state["pid"])
    procs.cleanup(tmp_path, everything=True)


def wait_for(check, seconds=40):
    """Waits until check() is true"""
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.2)
    return False


def kinds():
    return [(e["kind"], e["run"]) for e in control.get_events(limit=50)["events"]]


def test_queue_tools(repo):
    a = control.queue_run("r1", 3, {"rounds": 60, "workers": 2}, "first", "r1 should finish")
    assert a["id"] == 1 and "--rounds 60" in a["command"] and a["about_minutes"] >= 0
    with pytest.raises(ValueError):
        control.queue_run("r1", 3, {"rounds": 60, "workers": 2}, "again")
    with pytest.raises(ValueError):
        control.queue_run("r2", 3, {"rounds": 60, "workers": 2, "shell": "rm -rf"}, "bad setting")
    with pytest.raises(ValueError):
        control.queue_run("r2", 3, {"rounds": 60, "workers": 99}, "too many workers")
    with pytest.raises(ValueError):
        control.queue_resume("nope")
    q = control.list_queue()
    assert not q["supervisor_running"] and [e["run"] for e in q["queue"]] == ["r1"]
    assert control.remove_queued(1)["removed"]["run"] == "r1" and control.list_queue()["queue"] == []
    with pytest.raises(ValueError):
        control.remove_queued(1)
    assert "MCP: queued" in (jobs.DIR / "supervisor.log").read_text()


def test_start_watch_and_stop_through_the_supervisor(repo):
    with pytest.raises(ValueError):
        control.start_supervisor()
    control.queue_run("long", 3, {"rounds": 4000, "workers": 2}, "plain")
    control.queue_run("short", 3, {"rounds": 40, "workers": 2}, "plain")
    with pytest.raises(ValueError):
        control.start_supervisor("tomorrow")
    started = control.start_supervisor()
    assert started["queue"] == 2
    assert wait_for(lambda: control.live_status().get("run", {}).get("round", 0) >= 30)
    with pytest.raises(ValueError):
        control.start_supervisor()
    live = control.live_status()
    assert live["supervisor_running"] and live["phase"] == "training" and live["queue"] == ["short"]
    assert live["run"]["run"] == "long" and live["run"]["rounds"] == 4000 and live["run"]["rules_firing"] == []
    assert len(live["run"]["last_rows"]) == 3 and {p["kind"] for p in live["processes"]} >= {"train", "worker"}
    with pytest.raises(ValueError):
        control.stop_run(" ")
    assert control.stop_run("testing the stop")["how"] == "asked the supervisor"
    assert wait_for(lambda: ("queue_empty", None) in kinds())
    assert wait_for(lambda: not control.supervisor_alive())
    assert ("killed", "long") in kinds() and ("finished", "short") in kinds()
    killed = [e for e in control.get_events(limit=50)["events"] if e["kind"] == "killed"][0]
    assert killed["why"] == "testing the stop"
    assert registry.load_summary("long")["status"] == "killed" and registry.load_summary("short")["status"] == "finished"
    first = control.get_events(limit=1)
    assert len(first["events"]) == 1 and first["more"] >= 3
    assert control.get_events(since=first["events"][0]["n"], limit=1)["events"][0]["n"] == 2
    live = control.live_status()
    assert not live["supervisor_running"] and "run" not in live and live["processes"] == []


def test_stop_without_a_supervisor_and_cleanup(repo):
    by_hand = subprocess.Popen([sys.executable, "-m", "layout.train", "--run", "manual", "--stage", "3",
                                "--rounds", "5000", "--note", "by hand"])
    assert wait_for(lambda: len(procs.ours()) >= 3)
    assert control.cleanup_processes(dry_run=False)["leftover_workers"] == [] and by_hand.poll() is None
    out = control.stop_run("started by hand, not needed")
    assert "directly" in out["how"] and len(out["stopped"]) >= 3
    by_hand.wait(timeout=10)
    assert procs.ours() == []

    leaver = subprocess.Popen([sys.executable, "-m", "layout.train", "--run", "leaver", "--stage", "3",
                               "--rounds", "5000", "--note", "orphan please"])
    leaver.wait(timeout=30)
    assert wait_for(lambda: len(control.cleanup_processes()["leftover_workers"]) == 2)
    assert len(procs.ours()) == 2
    assert len(control.cleanup_processes(dry_run=False)["leftover_workers"]) == 2
    assert procs.ours() == []
