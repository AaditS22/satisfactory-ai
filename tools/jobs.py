import argparse
import json
import sys
import time
from pathlib import Path

from tools import registry

DIR = Path("runs/supervisor")
SETTINGS = {"rounds": int, "games": int, "workers": int, "T": int, "minibatch": int, "seed": int, "device": str,
            "eval_every": int, "images_every": int, "lr": float, "init": str, "ent_coef": float, "solvable": bool,
            "overlap": bool, "shaping": float, "copies": int, "groups": int, "broken_margin": float,
            "fixed_scale": bool, "scale": float, "cost_map": bool}
FROM_CONFIG = {"n_games": "games", "set_scale": "scale"}
NOT_ON_RESUME = ("init", "scale", "device")

def flag(name):
    """The command-line flag of a setting, e.g. ent_coef -> --ent-coef"""
    return "--" + name.replace("_", "-")

def run_folder(stage, run):
    """The folder training writes for a run"""
    return Path(f"runs/stage{stage}/{run}")

def load():
    """The queue file: the waiting entries and the next free id"""
    return registry.read_json(DIR / "queue.json", {"next": 1, "entries": []})

def save(queue):
    """Writes the queue file"""
    registry.write_json(DIR / "queue.json", queue, indent=1)

def clean(settings):
    """Checks every setting's name and type and gives them back as plain values"""
    out = {}
    for k, v in settings.items():
        if k not in SETTINGS:
            raise ValueError(f"unknown setting {k!r}; allowed: {sorted(SETTINGS)}")
        kind = SETTINGS[k]
        if v is None or (kind is bool and v is False):
            continue
        if kind is bool and v is not True:
            raise ValueError(f"{k} must be true or false")
        if kind is int and (isinstance(v, bool) or not isinstance(v, int)):
            raise ValueError(f"{k} must be a whole number")
        if kind is float and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise ValueError(f"{k} must be a number")
        if kind is str and not isinstance(v, str):
            raise ValueError(f"{k} must be text")
        out[k] = float(v) if kind is float else v
    return out

def check_limits(stage, settings, sup):
    """The limits every run must respect, new or resumed"""
    if stage not in (1, 2, 3, 4, 5):
        raise ValueError("stage must be 1 to 5")
    if not 1 <= settings.get("rounds", 50) <= sup["max_rounds"]:
        raise ValueError(f"rounds must be between 1 and {sup['max_rounds']}")
    if not 1 <= settings.get("workers", 24) <= sup["max_workers"]:
        raise ValueError(f"workers must be given and at most {sup['max_workers']} (more needs too much RAM)")
    for k in ("games", "copies", "groups", "minibatch", "eval_every", "T"):
        if k in settings and settings[k] < (0 if k == "copies" else 1):
            raise ValueError(f"{k} is too small")

def new_entry(run, stage, settings, note, prereg="", sup=None):
    """A checked queue entry for a new run"""
    sup = sup or registry.load_config()["supervisor"]
    registry.check_name(run)
    settings = clean(settings)
    check_limits(stage, settings, sup)
    if not note.strip():
        raise ValueError("a new run needs a note saying what it changes")
    if (run_folder(stage, run) / "last.pt").exists():
        raise ValueError(f"{run} already exists in runs/stage{stage}/: pick a new name, or queue a resume")
    if settings.get("scale") is not None and not settings.get("fixed_scale"):
        raise ValueError("scale needs fixed_scale")
    if settings.get("fixed_scale") and not settings.get("copies"):
        raise ValueError("fixed_scale needs copies")
    if "init" in settings:
        init = Path(settings["init"]).resolve()
        if Path.cwd().resolve() not in init.parents or init.suffix != ".pt" or not init.exists():
            raise ValueError(f"init must be an existing .pt file inside the repo, not {settings['init']!r}")
        settings["init"] = init.relative_to(Path.cwd().resolve()).as_posix()
    return {"run": run, "stage": stage, "resume": False, "settings": settings, "note": note.strip(), "prereg": prereg}

def resume_entry(run, stage=None, sup=None):
    """A checked queue entry that continues a run with exactly its own settings (read from its run.json)"""
    sup = sup or registry.load_config()["supervisor"]
    registry.check_name(run)
    stages = [stage] if stage else [1, 2, 3, 4, 5]
    folders = [run_folder(s, run) for s in stages if (run_folder(s, run) / "last.pt").exists()]
    if not folders:
        raise ValueError(f"no run {run} with a last.pt to resume")
    info = registry.read_json(folders[0] / "run.json")
    if not info or "config" not in info:
        raise ValueError(f"{run} has no run.json, so its settings are unknown")
    config = {FROM_CONFIG.get(k, k): v for k, v in info["config"].items()}
    settings = clean({k: v for k, v in config.items() if k in SETTINGS and k not in NOT_ON_RESUME})
    check_limits(config["stage"], settings, sup)
    return {"run": run, "stage": config["stage"], "resume": True, "settings": settings, "note": "", "prereg": ""}

def command(entry):
    """The exact training command of a queue entry, as a list of words"""
    argv = [sys.executable, "-m", "layout.train", "--run", entry["run"], "--stage", str(entry["stage"])]
    for k, v in entry["settings"].items():
        argv += [flag(k)] if v is True else [flag(k), str(v)]
    return argv + (["--resume"] if entry["resume"] else ["--note", entry["note"]])

def add(entry, sup=None):
    """Puts a checked entry at the end of the queue"""
    sup = sup or registry.load_config()["supervisor"]
    queue = load()
    if len(queue["entries"]) >= sup["max_queue"]:
        raise ValueError(f"the queue is full ({sup['max_queue']} runs)")
    if any(e["run"] == entry["run"] for e in queue["entries"]):
        raise ValueError(f"{entry['run']} is already in the queue")
    entry = {"id": queue["next"], **entry, "added": time.strftime("%Y-%m-%d %H:%M")}
    queue["entries"].append(entry)
    queue["next"] += 1
    save(queue)
    return entry

def remove(entry_id):
    """Takes an entry out of the queue and gives it back (None if it is not there)"""
    queue = load()
    found = next((e for e in queue["entries"] if e["id"] == entry_id), None)
    if found:
        queue["entries"].remove(found)
        save(queue)
    return found

def text(entry):
    """One queue entry as a line of text"""
    words = [w if " " not in w else f'"{w}"' for w in command(entry)[1:]]
    return f"#{entry.get('id', '-')}  {'resume' if entry['resume'] else 'new'}  python {' '.join(words)}"

def train_parser():
    """A parser that takes the same flags as layout.train"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run", required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--prereg", default="", help="what result will mean what, written before the run")
    p.add_argument("--prereg-file", default=None, help="a text file holding the pre-registration")
    for k, kind in SETTINGS.items():
        if kind is bool:
            p.add_argument(flag(k), action="store_true", default=None)
        else:
            p.add_argument(flag(k), type=kind, default=None)
    return p

def main():
    p = argparse.ArgumentParser(description="the queue of training runs the supervisor works through")
    sub = p.add_subparsers(dest="what", required=True)
    sub.add_parser("add", parents=[train_parser()], help="queue a new run: the same flags as layout.train")
    r = sub.add_parser("resume", help="queue the rest of a run that stopped early")
    r.add_argument("run")
    sub.add_parser("list", help="show the queue")
    d = sub.add_parser("remove", help="take an entry out of the queue")
    d.add_argument("id", type=int)
    args = p.parse_args()
    try:
        if args.what == "add":
            prereg = Path(args.prereg_file).read_text(encoding="utf-8") if args.prereg_file else args.prereg
            settings = {k: getattr(args, k) for k in SETTINGS}
            print("queued " + text(add(new_entry(args.run, args.stage, settings, args.note, prereg))))
        elif args.what == "resume":
            print("queued " + text(add(resume_entry(args.run))))
        elif args.what == "remove":
            gone = remove(args.id)
            print("removed " + text(gone) if gone else f"no entry #{args.id} in the queue")
        else:
            entries = load()["entries"]
            print("\n".join(text(e) for e in entries) if entries else "the queue is empty")
    except ValueError as e:
        p.error(str(e))

if __name__ == "__main__":
    main()
