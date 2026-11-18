import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from tools import registry, jobs, procs, rules, digest
from tools.curve import read_rows, avg

DIR = jobs.DIR

def stamp():
    """The time now as text"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(text):
    """Adds one line to the supervisor's log and prints it"""
    DIR.mkdir(parents=True, exist_ok=True)
    line = f"{stamp()}  {text}"
    with open(DIR / "supervisor.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)

def event(cfg, kind, run=None, **data):
    """Writes one numbered event file (and the log line for it) so a person or an agent can react later"""
    folder = DIR / "events"
    folder.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[:4]) for p in folder.glob("[0-9][0-9][0-9][0-9]_*.json")], default=0) + 1
    path = folder / f"{n:04d}_{kind}{'_' + run if run else ''}.json"
    registry.write_json(path, {"n": n, "time": stamp(), "kind": kind, "run": run, **data}, indent=1)
    log(f"EVENT {path.name}  {data.get('why', '')}")
    hook = cfg["supervisor"].get("on_event")
    if hook:
        try:
            subprocess.Popen(list(hook) + [str(path)])
        except OSError as e:
            log(f"on_event command failed: {e}")
    return path

def write_state(**fields):
    """Saves what the supervisor is doing right now (read by --status and later by the MCP server)"""
    old = registry.read_json(DIR / "state.json", {})
    state = {**old, **fields, "pid": os.getpid(), "heartbeat": stamp()}
    registry.write_json(DIR / "state.json", state, indent=1)
    return state

def other_supervisor():
    """The PID of another supervisor that is still running, or None"""
    pid = registry.read_json(DIR / "state.json", {}).get("pid")
    try:
        if pid and pid != os.getpid() and "tools.supervisor" in " ".join(psutil.Process(pid).cmdline()):
            return pid
    except psutil.Error:
        pass
    return None

def parse_deadline(text, now=None):
    """Turns '09:00' (the next time the clock shows it) or '2026-11-17 09:00' into a time"""
    now = now or datetime.datetime.now()
    if not text:
        return None
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        t = datetime.datetime.strptime(text, "%H:%M").time()
        day = datetime.datetime.combine(now.date(), t)
        return day if day > now else day + datetime.timedelta(days=1)

def sec_per_round(stage, sup):
    """Seconds per training round: measured from this stage's runs in the registry, else the config's guess"""
    seen = []
    for row in registry.load_index():
        train = (registry.load_summary(row["run"]) or {}).get("training") if row["stage"] == stage else None
        seen += [b["round_s"] for b in (train or {}).get("train_blocks", []) if b.get("round_s")]
    return sum(seen) / len(seen) if seen else sup["sec_per_round"]

def rounds_left(entry):
    """How many rounds a queue entry still has to train"""
    total = entry["settings"].get("stop_at") or entry["settings"].get("rounds", 50)
    if not entry["resume"]:
        return total
    rows = read_rows(jobs.run_folder(entry["stage"], entry["run"]) / "log.csv")
    return max(0, total - (rows[-1]["round"] + 1 if rows else 0))

def estimate_s(entry, sup):
    """A guess of how long a queue entry takes: training, its evals and the best_of at the end"""
    return rounds_left(entry) * sec_per_round(entry["stage"], sup) * sup["eval_overhead"] + sup["best_of_s"]

def save_patch(folder):
    """Saves every uncommitted change (new untracked files too) as code.patch, without touching git's own index"""
    try:
        git_dir = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, text=True).stdout.strip()
        if not git_dir:
            return "not a git repo: no patch saved"
        tmp = Path(git_dir) / "supervisor_index"
        if (Path(git_dir) / "index").exists():
            shutil.copyfile(Path(git_dir) / "index", tmp)
        env = {**os.environ, "GIT_INDEX_FILE": str(tmp.resolve())}
        subprocess.run(["git", "add", "-A"], env=env, capture_output=True, check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], env=env, capture_output=True, check=True)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        tmp.unlink(missing_ok=True)
        (folder / "code.patch").write_bytes(f"# changes on top of commit {head}\n".encode() + diff.stdout)
        return f"saved code.patch ({len(diff.stdout)} bytes on top of {head})"
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not save a patch: {e}"

def stop_request():
    """Reads and removes a stop request left by --stop (or later by the MCP server)"""
    path = DIR / "stop.json"
    req = registry.read_json(path)
    if req is not None:
        path.unlink(missing_ok=True)
    return req

def start(argv, out_path):
    """Starts a program with its output going to a file"""
    out = open(out_path, "a", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})

def tail(path, n=15):
    """The last lines of a text file"""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []

def progress(rows, evals, entry, sup):
    """The short summary written at a milestone"""
    last = rows[-1]["round"]
    total = entry["settings"].get("stop_at") or entry["settings"].get("rounds", 50)
    pace = avg(rows, "round_s", last - 49, last) or sup["sec_per_round"]
    return {"round": last, "rounds": total,
            "entropy": {"rounds_0_4": avg(rows, "entropy", 0, 4), "last_10": avg(rows, "entropy", last - 9, last)},
            "valid_score": {"rounds_0_4": avg(rows, "valid_score", 0, 4),
                            "last_10": avg(rows, "valid_score", last - 9, last)},
            "broken_last_20": avg(rows, "broken", last - 19, last),
            "kl_last_10": avg(rows, "approx_kl", last - 9, last),
            "eval": {"first": evals[0]["solv_score"] if evals else None,
                     "last_50_rounds": avg(evals, "solv_score", last - 49, last),
                     "note": "greedy eval dips and hovers on stage 3 even in good runs: not a kill rule"},
            "minutes_left": round((total - last - 1) * pace * sup["eval_overhead"] / 60),
            "free_ram_gb": round(procs.free_ram_gb(), 1), "gpu": procs.gpu()}

def watch(proc, entry, cfg):
    """Watches a training run until it ends or a kill rule fires; gives ('ended', code) or ('killed', rule, why)"""
    sup, run = cfg["supervisor"], entry["run"]
    folder = jobs.run_folder(entry["stage"], run)
    before = len(read_rows(folder / "log.csv"))
    done = {m for m in sup["milestones"] if before > m}
    warned, size, changed = set(), -1, time.time()
    while True:
        code = proc.poll()
        if code is not None:
            return ("ended", code)
        time.sleep(sup["poll_s"])
        rows, evals = read_rows(folder / "log.csv"), read_rows(folder / "eval.csv")
        new_size = (folder / "log.csv").stat().st_size if (folder / "log.csv").exists() else 0
        if new_size != size:
            size, changed = new_size, time.time()
        write_state(phase="training", run=run, stage=entry["stage"], train_pid=proc.pid,
                    round=rows[-1]["round"] if rows else None, rounds=entry["settings"].get("rounds", 50))
        req = stop_request()
        if req:
            return ("killed", "stop_all" if req.get("what") == "all" else "stop", req.get("reason") or "stop requested")
        tripped = rules.check(rows, evals, cfg["flags"], sup)
        tripped += [x for x in (rules.hang(time.time() - changed, len(rows) > before, sup),
                                rules.low_ram(procs.free_ram_gb(), sup)) if x]
        for rule, why in tripped:
            mode = sup["rules"].get(rule, "kill")
            if mode == "kill":
                return ("killed", rule, why)
            if mode == "flag" and rule not in warned:
                warned.add(rule)
                event(cfg, "warning", run, rule=rule, why=why)
        for m in sup["milestones"]:
            if rows and rows[-1]["round"] >= m and m not in done:
                done.add(m)
                event(cfg, "milestone", run, why=f"round {rows[-1]['round']} of {entry['settings'].get('rounds', 50)}",
                      milestone=m, **progress(rows, evals, entry, sup))

def run_best_of(entry, cfg):
    """Runs best_of on the run's last.pt and digests the result into the registry; gives the summary or None"""
    sup, run, stage = cfg["supervisor"], entry["run"], entry["stage"]
    folder = jobs.run_folder(stage, run)
    plans = DIR / "plans" / f"{run}.json"
    plans.parent.mkdir(parents=True, exist_ok=True)
    plans.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "layout.best_of", str(folder / "last.pt"), "--stage", str(stage),
            "--k", str(sup["best_of_k"]), "--save", str(plans)]
    log(f"best_of: {' '.join(argv[1:])}")
    proc, t0 = start(argv, folder / "best_of.out"), time.time()
    write_state(phase="best_of", run=run, stage=stage, train_pid=proc.pid)
    while proc.poll() is None:
        time.sleep(sup["poll_s"])
        write_state(phase="best_of")
        why = rules.low_ram(procs.free_ram_gb(), sup)
        why = why[1] if why else (stop_request() or {}).get("reason")
        if time.time() - t0 > sup["best_of_timeout_s"]:
            why = f"best_of ran longer than {sup['best_of_timeout_s'] / 60:.0f} min"
        if why:
            procs.kill_tree(proc.pid)
            event(cfg, "best_of_failed", run, why=why)
            return None
    if proc.returncode != 0 or not plans.exists():
        event(cfg, "best_of_failed", run, why=f"best_of ended with code {proc.returncode}",
              output=tail(folder / "best_of.out"))
        return None
    write_state(phase="digest")
    meta = {"plans": plans.name, "k": sup["best_of_k"], "seed": 0, "checkpoint": "last.pt", "note": entry["note"],
            "digested": time.strftime("%Y-%m-%d %H:%M")}
    try:
        summary = digest.digest(run, registry.read_json(plans), meta)
    except Exception as e:
        event(cfg, "digest_failed", run, why=f"{type(e).__name__}: {e}", plans=str(plans))
        return None
    if not sup["keep_plans"]:
        plans.unlink(missing_ok=True)
    return summary

def headline(s):
    """The few numbers of a finished run that go into its event file"""
    spec = s["vs"]["specialist"]
    return {"greedy": s["greedy"]["mean"], "greedy_broken": s["greedy"]["broken"], "best": s["best"]["mean"],
            "best_broken": s["best"]["broken"], "vs_B": s["vs"]["B"]["best"]["diff"],
            "specialist": spec.get("run"), "vs_specialist": spec.get("best", {}).get("diff"),
            "verdict": spec.get("best", {}).get("verdict"), "flags": s["flags"]}

def prepare(entry, cfg):
    """The checks before a run: leftover workers, other trainings, free RAM; gives a reason to refuse or None"""
    sup = cfg["supervisor"]
    gone = procs.cleanup()
    if gone:
        log(f"stopped {len(gone)} leftover worker(s) from earlier runs: pids {[g['pid'] for g in gone]}")
    busy = procs.active()
    if busy:
        return f"another training is already running in this repo (pids {[b['pid'] for b in busy]})"
    free = procs.free_ram_gb()
    if free < sup["min_free_ram_gb"]:
        time.sleep(min(30, sup["poll_s"]))
        free = procs.free_ram_gb()
    if free < sup["min_free_ram_gb"]:
        return f"only {free:.1f} GB of RAM free, {sup['min_free_ram_gb']} GB needed to start a run"
    log(f"free RAM {free:.1f} GB, GPU {procs.gpu()}")
    return None

def recheck(entry, sup):
    """Checks a queue entry again right before it starts (things may have changed since it was queued)"""
    if entry["resume"]:
        return {**entry, **jobs.resume_entry(entry["run"], entry["stage"], sup)}
    return {**entry, **jobs.new_entry(entry["run"], entry["stage"], entry["settings"], entry["note"],
                                      entry.get("prereg", ""), sup)}

def run_one(entry, cfg):
    """Trains one queue entry from start to registry entry; gives False if the supervisor should stop afterwards"""
    sup, run, stage = cfg["supervisor"], entry["run"], entry["stage"]
    folder = jobs.run_folder(stage, run)
    folder.mkdir(parents=True, exist_ok=True)
    log(save_patch(folder))
    if entry.get("prereg") and not (folder / "prereg.json").exists():
        registry.write_json(folder / "prereg.json", {"written": stamp(), "text": entry["prereg"], "entry": entry}, indent=1)
    argv = jobs.command(entry)
    log(f"starting {run}: {' '.join(argv[1:])}")
    proc = start(argv, folder / "train.out")
    minutes = round(estimate_s(entry, sup) / 60)
    event(cfg, "started", run, why=f"about {minutes} min with best_of", command=argv[1:], pid=proc.pid, minutes=minutes)
    try:
        result = watch(proc, entry, cfg)
    except KeyboardInterrupt:
        result = ("killed", "stop_all", "Ctrl+C in the supervisor window")
    if result[0] == "killed":
        _, rule, why = result
        stopped = procs.kill_tree(proc.pid)
        proc.wait()
        digest.digest_training(run, stage, "killed", f"{rule}: {why}")
        event(cfg, "killed", run, rule=rule, why=why, stopped_pids=stopped)
        return rule != "stop_all"
    rows = read_rows(folder / "log.csv")
    total = entry["settings"].get("stop_at") or entry["settings"].get("rounds", 50)
    if result[1] != 0 or not rows or rows[-1]["round"] < total - 1:
        why = f"training ended with code {result[1]} at round {rows[-1]['round'] if rows else None} of {total}"
        procs.kill_tree(proc.pid)
        digest.digest_training(run, stage, "crashed", why)
        event(cfg, "crashed", run, why=why, output=tail(folder / "train.out"))
        return True
    log(f"{run} trained all {total} rounds")
    procs.cleanup()
    summary = run_best_of(entry, cfg)
    if summary is None:
        digest.digest_training(run, stage, "trained", "training finished but best_of or the digest failed")
        return True
    h = headline(summary)
    event(cfg, "finished", run, why=f"greedy {h['greedy']} ({h['greedy_broken']} broken), best {h['best']} "
                                    f"({h['best_broken']} broken), vs specialist {h['verdict']}", **h)
    return True

def pick(cfg, deadline):
    """The first queue entry that fits before the deadline, plus the ones that do not fit"""
    late = []
    for entry in jobs.load()["entries"]:
        end = datetime.datetime.now() + datetime.timedelta(seconds=estimate_s(entry, cfg["supervisor"]))
        if deadline is None or end <= deadline:
            return entry, late
        late.append({"run": entry["run"], "would_end": end.strftime("%Y-%m-%d %H:%M")})
    return None, late

def supervise(cfg, deadline=None):
    """Works through the queue until it is empty, nothing fits before the deadline, or it is told to stop"""
    DIR.mkdir(parents=True, exist_ok=True)
    stop_request()
    write_state(started=stamp(), deadline=str(deadline) if deadline else None, phase="starting", run=None)
    log(f"supervisor started (pid {os.getpid()}, deadline {deadline})")
    while True:
        entry, late = pick(cfg, deadline)
        if entry is None:
            if late:
                event(cfg, "deadline", why=f"no queued run can finish by {deadline}", skipped=late)
            event(cfg, "queue_empty", why="nothing left to run", waiting=[x["run"] for x in late])
            break
        try:
            entry = recheck(entry, cfg["supervisor"])
        except ValueError as e:
            jobs.remove(entry["id"])
            event(cfg, "refused", entry["run"], why=f"dropped from the queue: {e}")
            continue
        why = prepare(entry, cfg)
        if why:
            event(cfg, "refused", entry["run"], why=why + "; the run stays in the queue and the supervisor stops")
            break
        jobs.remove(entry["id"])
        try:
            if not run_one(entry, cfg):
                break
        except KeyboardInterrupt:
            gone = procs.cleanup(everything=True)
            event(cfg, "killed", entry["run"], rule="stop_all", why="Ctrl+C in the supervisor window",
                  stopped_pids=[g["pid"] for g in gone])
            break
        except Exception:
            gone = procs.cleanup(everything=True)
            event(cfg, "supervisor_error", entry["run"], why="the supervisor itself hit an error and stopped",
                  error=traceback.format_exc().splitlines()[-12:], stopped_pids=[g["pid"] for g in gone])
            break
    procs.cleanup()
    write_state(phase="stopped", run=None, train_pid=None)
    log("supervisor stopped")

def status():
    """Prints what the supervisor is doing, the queue and the latest events"""
    state = registry.read_json(DIR / "state.json", {})
    alive = bool(state.get("pid")) and psutil.pid_exists(state["pid"]) and state.get("phase") != "stopped"
    print(f"supervisor: {'running' if alive else 'not running'}  {state}")
    print(f"free RAM {procs.free_ram_gb():.1f} GB   GPU {procs.gpu()}")
    for e in jobs.load()["entries"]:
        print("queued " + jobs.text(e))
    for p in sorted((DIR / "events").glob("*.json"))[-8:]:
        e = registry.read_json(p)
        print(f"{e['time']}  {p.name}  {e.get('why', '')}")

def dry_run(cfg, deadline):
    """Prints what the supervisor would do with the queue, without starting anything"""
    now = datetime.datetime.now()
    for e in jobs.load()["entries"]:
        est = estimate_s(e, cfg["supervisor"])
        end = now + datetime.timedelta(seconds=est)
        fits = deadline is None or end <= deadline
        print(f"{jobs.text(e)}\n   about {est / 3600:.1f} h, would end {end:%H:%M}" + ("" if fits else "  DOES NOT FIT the deadline"))
        now = end if fits else now
    print(f"free RAM {procs.free_ram_gb():.1f} GB (needs {cfg['supervisor']['min_free_ram_gb']})   "
          f"leftover workers {len([f for f in procs.ours() if f['orphan']])}   "
          f"trainings running {len(procs.active())}")

def main():
    p = argparse.ArgumentParser(description="works through the queue of training runs, no AI involved")
    p.add_argument("--deadline", default=None, help="do not start a run that would not finish by then: 09:00 or '2026-11-17 09:00'")
    p.add_argument("--dry-run", action="store_true", help="show what would happen, start nothing")
    p.add_argument("--status", action="store_true", help="show the supervisor's state, the queue and the last events")
    p.add_argument("--stop", metavar="REASON", help="stop the current run (the supervisor goes on with the queue)")
    p.add_argument("--stop-all", metavar="REASON", help="stop the current run and the supervisor")
    args = p.parse_args()
    cfg = registry.load_config()
    if args.status:
        return status()
    if args.stop or args.stop_all:
        registry.write_json(DIR / "stop.json", {"what": "all" if args.stop_all else "run",
                                                "reason": args.stop_all or args.stop, "time": stamp()})
        return print(f"stop requested; the supervisor acts within {cfg['supervisor']['poll_s']} s")
    deadline = parse_deadline(args.deadline)
    if args.dry_run:
        return dry_run(cfg, deadline)
    if other_supervisor():
        p.error(f"a supervisor is already running (pid {other_supervisor()})")
    supervise(cfg, deadline)

if __name__ == "__main__":
    main()
