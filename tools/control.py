import os
import subprocess
import sys

import psutil

from tools import registry, jobs, procs, rules, supervisor
from tools.curve import read_rows

DIR = jobs.DIR

def note(text):
    """Writes what was done through the MCP server into the supervisor's log"""
    supervisor.log(f"MCP: {text}")

def supervisor_alive():
    """True if a supervisor is running right now"""
    state = registry.read_json(DIR / "state.json", {})
    try:
        return bool(state.get("pid")) and state.get("phase") != "stopped" \
            and "tools.supervisor" in " ".join(psutil.Process(state["pid"]).cmdline())
    except psutil.Error:
        return False

def with_command(entry, sup):
    """A queue entry with its exact command and a time estimate added"""
    return {**entry, "command": jobs.text(entry), "about_minutes": round(supervisor.estimate_s(entry, sup) / 60)}

def queue_run(run, stage, settings, note_text, prereg=""):
    """Checks and queues a new training run"""
    sup = registry.load_config()["supervisor"]
    entry = jobs.add(jobs.new_entry(run, stage, settings, note_text, prereg, sup), sup)
    note(f"queued {jobs.text(entry)}")
    return with_command(entry, sup)

def queue_resume(run):
    """Checks and queues the rest of a run that stopped early"""
    sup = registry.load_config()["supervisor"]
    entry = jobs.add(jobs.resume_entry(run, sup=sup), sup)
    note(f"queued {jobs.text(entry)}")
    return with_command(entry, sup)

def list_queue():
    """The waiting runs, in the order the supervisor will take them"""
    sup = registry.load_config()["supervisor"]
    return {"supervisor_running": supervisor_alive(), "queue": [with_command(e, sup) for e in jobs.load()["entries"]]}

def remove_queued(entry_id):
    """Takes one waiting run out of the queue"""
    gone = jobs.remove(entry_id)
    if gone is None:
        raise ValueError(f"no entry #{entry_id} in the queue")
    note(f"removed from the queue: {jobs.text(gone)}")
    return {"removed": gone}

def stop_run(reason, stop_supervisor=False):
    """Stops the training that is running now: through the supervisor if there is one, else directly"""
    if not reason.strip():
        raise ValueError("give a reason: it is written to the log and the registry")
    if supervisor_alive():
        registry.write_json(DIR / "stop.json", {"what": "all" if stop_supervisor else "run", "reason": reason,
                                                "time": supervisor.stamp()})
        note(f"stop requested ({'run and supervisor' if stop_supervisor else 'current run'}): {reason}")
        return {"how": "asked the supervisor", "acts_within_s": registry.load_config()["supervisor"]["poll_s"],
                "run": registry.read_json(DIR / "state.json", {}).get("run")}
    gone = procs.cleanup(everything=True)
    note(f"no supervisor running; stopped {len(gone)} training process(es) directly: {reason}")
    return {"how": "no supervisor running, stopped this repo's training processes directly", "stopped": gone}

def cleanup_processes(dry_run=True):
    """Finds (and unless dry_run, stops) leftover workers whose training program is gone"""
    found = procs.cleanup(dry_run=dry_run)
    if found and not dry_run:
        note(f"stopped {len(found)} leftover worker(s): pids {[f['pid'] for f in found]}")
    return {"dry_run": dry_run, "leftover_workers": found, "free_ram_gb": round(procs.free_ram_gb(), 1)}

def start_supervisor(deadline=""):
    """Starts the supervisor in the background if none is running and the queue is not empty"""
    if supervisor_alive():
        raise ValueError("a supervisor is already running")
    if not jobs.load()["entries"]:
        raise ValueError("the queue is empty: queue a run first")
    try:
        when = supervisor.parse_deadline(deadline)
    except ValueError:
        raise ValueError("deadline must look like 09:00 or 2026-11-17 09:00 (or be empty)")
    argv = [sys.executable, "-m", "tools.supervisor"] + (["--deadline", deadline] if deadline else [])
    DIR.mkdir(parents=True, exist_ok=True)
    out = open(DIR / "supervisor.out", "a", encoding="utf-8")
    proc = launch(argv, out)
    note(f"started the supervisor (pid {proc.pid}, deadline {when})")
    return {"pid": proc.pid, "deadline": str(when) if when else None, "queue": len(jobs.load()["entries"])}

def launch(argv, out):
    """Starts a program that keeps running after the MCP server (and Claude's app) closes"""
    if os.name != "nt":
        return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags)

def short_row(r):
    """The few numbers of one log row worth showing live"""
    keys = ("round", "valid_score", "entropy", "broken", "approx_kl", "push", "round_s")
    return {k: round(r[k], 4) for k in keys if k in r and r[k] == r[k]}

def live_status():
    """What is happening right now: supervisor, current run with its latest numbers, memory, queue, last events"""
    cfg = registry.load_config()
    sup = cfg["supervisor"]
    state = registry.read_json(DIR / "state.json", {})
    alive = supervisor_alive()
    out = {"supervisor_running": alive, "phase": state.get("phase") if alive else None,
           "deadline": state.get("deadline") if alive else None, "heartbeat": state.get("heartbeat"),
           "free_ram_gb": round(procs.free_ram_gb(), 1), "gpu": procs.gpu(),
           "processes": [{k: p[k] for k in ("pid", "kind", "ram_gb", "orphan")} for p in procs.ours()],
           "queue": [e["run"] for e in jobs.load()["entries"]],
           "last_events": [p.name for p in sorted((DIR / "events").glob("*.json"))[-6:]]}
    if alive and state.get("run"):
        folder = jobs.run_folder(state["stage"], state["run"])
        rows, evals = read_rows(folder / "log.csv"), read_rows(folder / "eval.csv")
        out["run"] = {"run": state["run"], "stage": state["stage"]}
        if rows:
            entry = {"settings": {"rounds": state.get("rounds") or rows[-1]["round"] + 1}}
            out["run"].update(supervisor.progress(rows, evals, entry, sup))
            out["run"]["rules_firing"] = [{"rule": r, "why": w, "mode": sup["rules"].get(r, "kill")}
                                          for r, w in rules.check(rows, evals, cfg["flags"], sup)]
            out["run"]["last_rows"] = [short_row(r) for r in rows[-3:]]
            out["run"]["last_evals"] = [{"round": e["round"], "solv_score": round(e["solv_score"], 4)} for e in evals[-5:]]
    return out

def get_events(since=0, limit=20):
    """The supervisor's event files with a number above 'since', oldest first"""
    files = [p for p in sorted((DIR / "events").glob("[0-9][0-9][0-9][0-9]_*.json")) if int(p.name[:4]) > since]
    return {"events": [registry.read_json(p) for p in files[:max(1, min(limit, 50))]],
            "more": max(0, len(files) - max(1, min(limit, 50)))}
