import json
import os
import re
import time
from pathlib import Path

ROOT = Path("runs/registry")
HERE = Path(__file__).parent

def read_json(path, default=None):
    """Reads a JSON file, or gives the default if it does not exist"""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, data, indent=None):
    """Writes a JSON file in one step, so a crash never leaves half a file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    for attempt in range(5):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)

def load_config():
    """The editable settings in tools/config.json"""
    return read_json(HERE / "config.json")

def specialists():
    """The current specialist run of every stage, from tools/specialists.json"""
    return {int(k): v for k, v in read_json(HERE / "specialists.json", {}).items()}

def check_name(run):
    """Stops run names that could not be a plain folder name"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run) or run.startswith("."):
        raise ValueError(f"bad run name {run!r}: use letters, digits, _ . - only")
    return run

def run_file(run, name):
    """Path of one file of a run's registry entry"""
    return ROOT / check_name(run) / name

def load_summary(run):
    """A run's compact summary, or None if the run is not in the registry"""
    return read_json(run_file(run, "summary.json"))

def load_tasks(run):
    """A run's per-task results, or None if the run is not in the registry"""
    return read_json(run_file(run, "tasks.json"))

def load_index():
    """One short row per run in the registry"""
    return read_json(ROOT / "index.json", [])

def all_runs():
    """Names of every run that has per-task results in the registry"""
    if not ROOT.exists():
        return []
    return sorted(p.parent.name for p in ROOT.glob("*/tasks.json"))

def index_row(s):
    """The short index row of one summary"""
    if "best" not in s:
        return {"run": s["run"], "stage": s["stage"], "status": s["status"], "digested": s["source"]["digested"],
                "k": None, "greedy": None, "greedy_broken": None, "best": None, "best_broken": None, "vs_A": None,
                "vs_B": None, "specialist": None, "vs_specialist": None, "flags": s["flags"]}
    spec = s["vs"]["specialist"]
    return {"run": s["run"], "stage": s["stage"], "status": s.get("status", "finished"),
            "digested": s["source"]["digested"], "k": s["source"]["k"],
            "greedy": s["greedy"]["mean"], "greedy_broken": s["greedy"]["broken"],
            "best": s["best"]["mean"], "best_broken": s["best"]["broken"],
            "vs_A": s["vs"]["A"]["best"]["diff"], "vs_B": s["vs"]["B"]["best"]["diff"],
            "specialist": spec.get("run"), "vs_specialist": spec["best"]["verdict"] if "best" in spec else None,
            "flags": s["flags"]}

def save_run(summary, tasks=None):
    """Stores a run's summary (and per-task results) and updates the index"""
    run = summary["run"]
    if tasks is not None:
        write_json(run_file(run, "tasks.json"), tasks)
    write_json(run_file(run, "summary.json"), summary, indent=1)
    rows = [r for r in load_index() if r["run"] != run] + [index_row(summary)]
    rows.sort(key=lambda r: (r["stage"], r["digested"], r["run"]))
    write_json(ROOT / "index.json", rows, indent=1)
