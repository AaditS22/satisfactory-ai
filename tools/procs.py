import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import psutil

def free_ram_gb():
    """Free memory in GB"""
    return psutil.virtual_memory().available / 2**30

def gpu():
    """GPU memory and load from nvidia-smi, or None if there is no such tool"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        used, total, load = [float(x) for x in out.strip().splitlines()[0].split(",")]
        return {"used_gb": round(used / 1024, 2), "total_gb": round(total / 1024, 2), "load": load / 100}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None

def classify(cmdline):
    """What kind of project process a command line is: 'train', 'best_of', 'worker' or None"""
    text = " ".join(cmdline)
    if "multiprocessing-fork" in text or "spawn_main" in text:
        return "worker"
    if "layout.train" in text or text.replace("\\", "/").find("layout/train.py") >= 0:
        return "train"
    if "layout.best_of" in text or text.replace("\\", "/").find("layout/best_of.py") >= 0:
        return "best_of"
    return None

def declared_parent(cmdline):
    """The parent PID a spawn worker's command line names (Windows writes parent_pid=...), or None"""
    m = re.search(r"parent_pid=(\d+)", " ".join(cmdline))
    return int(m.group(1)) if m else None

def describe(p, repo):
    """One process as a dict if it is a training process of this repo, else None"""
    try:
        cmd = p.cmdline()
        kind = classify(cmd)
        if kind is None or "python" not in p.name().lower() or Path(p.cwd()).resolve() != repo:
            return None
        parent = declared_parent(cmd) or p.ppid()
        return {"pid": p.pid, "kind": kind, "parent": parent, "started": p.create_time(),
                "ram_gb": round(p.memory_info().rss / 2**30, 2), "orphan": kind == "worker" and orphaned(p, parent)}
    except (psutil.Error, OSError):
        return None

def orphaned(p, parent):
    """A worker whose parent program is gone (or whose parent PID now belongs to something else)"""
    try:
        boss = psutil.Process(parent)
        return "python" not in boss.name().lower() or boss.create_time() > p.create_time() \
            or boss.status() == psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return True

def ours(repo="."):
    """Every training, best_of and worker process that was started inside this repo folder"""
    repo = Path(repo).resolve()
    found = [describe(p, repo) for p in psutil.process_iter() if p.pid != os.getpid()]
    return sorted([f for f in found if f], key=lambda f: f["started"])

def tree(pid):
    """A process and everything below it, children first"""
    try:
        top = psutil.Process(pid)
        return top.children(recursive=True)[::-1] + [top]
    except psutil.Error:
        return []

def kill(procs, wait=5):
    """Stops processes: asks nicely first, then forces the ones still alive; gives the PIDs it stopped"""
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=wait)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=wait)
    return [p.pid for p in procs]

def kill_tree(pid):
    """Stops a process and its whole tree, plus spawn workers that name it as their parent"""
    procs = tree(pid)
    seen = {p.pid for p in procs}
    for p in psutil.process_iter():
        try:
            if p.pid not in seen and declared_parent(p.cmdline()) in seen:
                procs.insert(0, p)
        except (psutil.Error, OSError):
            pass
    return kill(procs)

def cleanup(repo=".", everything=False, dry_run=False):
    """Stops leftover workers whose parent is gone (or, with everything, every training process of this repo)"""
    targets = [f for f in ours(repo) if everything or f["orphan"]]
    if not dry_run:
        procs = []
        for f in targets:
            try:
                p = psutil.Process(f["pid"])
                if p.create_time() == f["started"]:
                    procs += [c for c in tree(f["pid"]) if c.pid not in {x.pid for x in procs}]
            except psutil.Error:
                pass
        kill(procs)
    return targets

def active(repo="."):
    """Training or best_of programs of this repo that are running right now"""
    return [f for f in ours(repo) if f["kind"] in ("train", "best_of")]

def main():
    p = argparse.ArgumentParser(description="list or stop this repo's training processes")
    p.add_argument("--kill", action="store_true", help="stop leftover workers whose parent is gone")
    p.add_argument("--all", action="store_true", help="with --kill: stop every training process of this repo")
    args = p.parse_args()
    found = cleanup(everything=args.all, dry_run=not args.kill) if args.kill else ours()
    print(f"free RAM {free_ram_gb():.1f} GB   GPU {gpu()}")
    for f in found:
        age = (time.time() - f["started"]) / 60
        print(f"{'stopped' if args.kill else 'found'} {f['kind']:8s} pid {f['pid']:6d}  parent {f['parent']:6d}  "
              f"{f['ram_gb']:.2f} GB  {age:.0f} min old" + ("  ORPHAN" if f["orphan"] else ""))
    if not found:
        print("nothing to stop" if args.kill else "no training processes of this repo are running")

if __name__ == "__main__":
    main()
