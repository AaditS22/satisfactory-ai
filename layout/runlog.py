import csv
from pathlib import Path

import numpy as np

PARTS = ["p_belt", "p_lifts", "p_crossings", "p_sprawl", "p_floors"]
GROUPS = {"score": "score", "valid_score": "score", "failed": "rates", "invalid": "rates", "routed": "rates",
          "fraction": "rates", "games": "games", "blocks": "games", "collect_s": "time", "update_s": "time",
          "gpu_gb": "time"}

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
    return out

class RunLog:
    """Writes one row per round to a CSV file"""
    def __init__(self, path, tensorboard=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = None
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.path.parent))

    def write(self, row):
        new = self.fields is None
        if new:
            self.fields = list(row)
        with open(self.path, "w" if new else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
        if self.tb is not None:
            for k, v in row.items():
                if k != "round":
                    self.tb.add_scalar(tag(k), v, row["round"])
            self.tb.flush()

    def close(self):
        if self.tb is not None:
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
            f"play {row['collect_s']:.1f} s  learn {row['update_s']:.1f} s")