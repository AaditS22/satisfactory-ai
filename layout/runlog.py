import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PARTS = ["p_belt", "p_lifts", "p_crossings", "p_sprawl", "p_floors"]
GROUPS = {"score": "score", "valid_score": "score", "failed": "rates", "invalid": "rates", "routed": "rates",
          "fraction": "rates", "games": "games", "blocks": "games", "collect_s": "time", "update_s": "time",
          "gpu_gb": "time", "play_s": "time", "wire": "penalty", "steps": "games", "groups": "games", "idle": "time",
          "flat": "rates"}

def run_dir(stage, run):
    """Finds the path to the folder of a run"""
    return f"runs/stage{stage}/{run}"

def git_state():
    """Gets the current state of the code from git"""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""
    return head + ("+changes" if head and dirty else "")

def save_run_info(folder, note, config, resumed_at=None):
    """Saves important info about a run to a json file"""
    path = Path(folder) / "run.json"
    now = time.strftime("%Y-%m-%d %H:%M")
    if resumed_at is not None and path.exists():
        info = json.loads(path.read_text())
        info.setdefault("resumed", []).append({"at_round": resumed_at, "time": now, "code": git_state(), "note": note})
    else:
        info = {"note": note, "started": now, "command": " ".join(sys.argv), "code": git_state(), "config": config,
                "resumed": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1))
    return info

def tag(key):
    if key in PARTS:
        return "penalty/" + key
    return GROUPS.get(key, "ppo") + "/" + key

def mean(xs):
    """Mean of a list, or nan if it is empty"""
    return float(np.mean(xs)) if len(xs) else float("nan")

def kind(info):
    """Finds what happened in a finished game: failed, invalid or valid"""
    if info["failed"]:
        return "failed"
    if not info["terms"]["passed"]:
        return "invalid"
    return "valid"

def round_stats(finished):
    """Summary of the games that ended in one round"""
    kinds = [kind(f) for f in finished]
    valid = [f for f, k in zip(finished, kinds) if k == "valid"]
    done = [f for f, k in zip(finished, kinds) if k != "failed"]
    out = {"games": len(finished),
           "score": mean([f["score"] for f in finished]),
           "failed": mean([k == "failed" for k in kinds]),
           "invalid": mean([k == "invalid" for k in kinds]),
           "valid_score": mean([f["score"] for f in valid]),
           "blocks": mean([f["blocks"] for f in finished])}
    for p in PARTS:
        out[p] = mean([f["terms"][p] for f in valid])
    out["routed"] = mean([f["terms"]["routed"] for f in done])
    out["fraction"] = mean([f["terms"]["fraction"] for f in done])
    out["wire"] = mean([f["wire_total"] for f in valid if "wire_total" in f])
    return out

class RunLog:
    """Writes one row per round to a CSV file"""

    def __init__(self, path, tensorboard=True, group=None, append=False, share=None):
        self.path = Path(path)
        self.group = group
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = None
        if append and self.path.exists():
            with open(self.path, newline="") as f:
                self.fields = next(csv.reader(f))
        self.tb = None
        self.owns_tb = False
        if share is not None:
            self.tb = share.tb
        elif tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.path.parent))
            self.owns_tb = True

    def write(self, row):
        new = self.fields is None
        if new:
            self.fields = list(row)
        with open(self.path, "w" if new else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow({k: float(f"{v:.5g}") if isinstance(v, float) else v for k, v in row.items()})
        if self.tb is not None:
            for k, v in row.items():
                if k != "round":
                    name = f"{self.group}/{k}" if self.group else tag(k)
                    self.tb.add_scalar(name, v, row["round"])
            self.tb.flush()

    def image(self, name, png, step):
        """Show a saved PNG in TensorBoard's Images tab"""
        if self.tb is not None:
            import matplotlib.image
            pixels = matplotlib.image.imread(str(png))[:, :, :3]
            self.tb.add_image(name, pixels, step, dataformats="HWC")
            self.tb.flush()

    def text(self, name, s):
        """Show a piece of text in TensorBoard's Text tab"""
        if self.tb is not None:
            self.tb.add_text(name, s, 0)
            self.tb.flush()

    def close(self):
        """Close the TensorBoard writer if this RunLog made it"""
        if self.owns_tb:
            self.tb.close()

def read_log(path):
    """Read a run's CSV back"""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def line(row):
    """The short line printed every round"""
    return (f"round {row['round']:4d}  games {row['games']:4d}  score {row['score']:.3f}  "
            f"valid {row['valid_score']:.3f}  failed {row['failed']:.2f}  invalid {row['invalid']:.2f}  "
            f"blocks {row['blocks']:.1f}  entropy {row['entropy']:.2f}  kl {row['approx_kl']:.4f}  "
            f"clip {row['clip_frac']:.3f}  vloss {row['value_loss']:.3f}  "
            f"wait {row['collect_s']:.1f} s  learn {row['update_s']:.1f} s"
            + (f"  (page took {row['play_s']:.1f} s)" if "play_s" in row else "")
            + (f"  steps {row['steps']}  idle {row['idle']:.0%}  flat {row['flat']:.0%}" if "idle" in row else ""))