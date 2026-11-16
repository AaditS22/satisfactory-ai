import hashlib
import json
from pathlib import Path

from tools import registry
from tools.registry import read_json, write_json, load_config
from tools.replay import replay

EVAL_PATH = Path("data/eval_set.json")

def eval_rows(stage=None, path=EVAL_PATH):
    """The fixed eval tasks (of one stage, or all of them)"""
    rows = json.loads(Path(path).read_text())
    for r in rows:
        if r["task"].get("alternates") is not None:
            r["task"]["alternates"] = frozenset(r["task"]["alternates"])
    return [r for r in rows if stage is None or r["stage"] == stage]

def stage_of(seeds, rows):
    """The one stage a plan file's seeds belong to"""
    by_seed = {r["seed"]: r["stage"] for r in rows}
    stages = {by_seed.get(s) for s in seeds}
    if None in stages or len(stages) != 1:
        raise ValueError(f"the seeds must all be eval tasks of one stage, found stages {sorted(map(str, stages))}")
    return stages.pop()

def fingerprint(path=EVAL_PATH):
    """Short hash of the eval set file, to notice when it changes"""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]

def base_row(r, big_block):
    """One eval task with A's and B's plans replayed through the real scorer"""
    t = r["task"]
    a, b = replay(t, r["A"]["plan"]), replay(t, r["B"]["plan"])
    return {"seed": r["seed"], "target": r["target"].replace("Desc_", "").replace("_C", ""), "rate": t["rate"],
            "width": t["width"], "depth": t["depth"], "solvable": r["A"]["score"] > 0,
            "big": r["A"]["score"] > 0 and max(a["blocks"]) >= big_block,
            "A": {**a, "saved": r["A"]["score"]}, "B": {**b, "saved": r["B"]["score"]}}

def build(stage, rows=None):
    """Replays A and B on every eval task of a stage"""
    rows = eval_rows(stage) if rows is None else rows
    big = load_config()["big_block"]
    return {"stage": stage, "eval_set": fingerprint(), "big_block": big, "tasks": [base_row(r, big) for r in rows]}

def baselines(stage):
    """A's and B's replayed results for a stage, replayed once and then read from the registry"""
    path = registry.ROOT / "baselines" / f"stage{stage}.json"
    base = read_json(path)
    if base is None or base["eval_set"] != fingerprint() or base["big_block"] != load_config()["big_block"]:
        print(f"replaying baselines A and B for stage {stage} (once, about half a minute)...", flush=True)
        base = build(stage)
        write_json(path, base)
    return base
