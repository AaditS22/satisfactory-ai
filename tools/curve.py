import csv
import math
from pathlib import Path

from tools.registry import read_json
from tools.stats import mean

TRAIN_KEYS = ["entropy", "valid_score", "broken", "push", "flat", "approx_kl", "clip_frac", "round_s"]
EVAL_KEYS = ["solv_score", "gate_A", "broken"]
WATCHED = ["score", "entropy", "approx_kl", "value_loss", "policy_loss"]

def number(text):
    """A CSV cell as a number (nan if it is empty or not a number)"""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")

def read_rows(path):
    """Reads a run's CSV as one dict per round; a round written twice (after a resume) keeps its later row"""
    path = Path(path)
    if not path.exists():
        return []
    by_round = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {k: number(v) for k, v in r.items() if k is not None}
            if not math.isnan(row.get("round", float("nan"))):
                row["round"] = int(row["round"])
                row["broken"] = row.get("failed", float("nan")) + row.get("invalid", float("nan"))
                row["round_s"] = row.get("collect_s", float("nan")) + row.get("update_s", float("nan"))
                by_round[row["round"]] = row
    return [by_round[r] for r in sorted(by_round)]

def avg(rows, key, lo=None, hi=None):
    """Mean of one column over rounds lo..hi (both included), skipping nan"""
    xs = [r[key] for r in rows if key in r and (lo is None or r["round"] >= lo) and (hi is None or r["round"] <= hi)
          and not math.isnan(r[key])]
    return mean(xs)

def blocks(rows, keys, total, n):
    """Means of the columns over n equal blocks of rounds"""
    size = max(1, math.ceil(total / n))
    out = []
    for lo in range(0, total, size):
        part = [r for r in rows if lo <= r["round"] < lo + size]
        if part:
            out.append({"rounds": f"{lo}-{min(lo + size, total) - 1}", **{k: avg(part, k) for k in keys}})
    return out

def bad_numbers(rows):
    """The first round where a watched number is nan or infinite, or None"""
    for r in rows:
        for k in WATCHED:
            if k in r and not math.isfinite(r[k]):
                return {"round": r["round"], "column": k}
    return None

def drift(rows, evals, cfg):
    """The drift fingerprint: a 10-round window with entropy up and valid score down vs rounds 0-4, eval not rising"""
    e0, v0 = avg(rows, "entropy", 0, 4), avg(rows, "valid_score", 0, 4)
    if e0 is None or v0 is None or not rows:
        return None
    first_eval = evals[0]["solv_score"] if evals else None
    for lo in range(10, rows[-1]["round"] - 8, 10):
        e, v = avg(rows, "entropy", lo, lo + 9), avg(rows, "valid_score", lo, lo + 9)
        if e is None or v is None or e - e0 < cfg["drift_entropy"] or v - v0 > -cfg["drift_valid"]:
            continue
        recent = avg(evals, "solv_score", max(1, lo - 40), lo + 9)
        if first_eval is None or recent is None or recent <= first_eval:
            return {"rounds": f"{lo}-{lo + 9}", "entropy": e - e0, "valid_score": v - v0,
                    "eval_start": first_eval, "eval_recent": recent}
    return None

def flags(rows, evals, planned, cfg):
    """The automatic warnings about a run's training curve"""
    out = []
    if not rows:
        return ["no training rows found"]
    last = rows[-1]["round"]
    if planned and last < planned - 1:
        out.append(f"incomplete run: stopped at round {last} of {planned}")
    bad = bad_numbers(rows)
    if bad:
        out.append(f"{bad['column']} is nan or infinite at round {bad['round']}")
    d = drift(rows, evals, cfg)
    if d:
        out.append(f"drift fingerprint at rounds {d['rounds']}: entropy {d['entropy']:+.2f}, "
                   f"valid score {d['valid_score']:+.3f} vs rounds 0-4, eval not rising")
    end = avg(rows, "entropy", last - 9, last)
    if end is not None and end < cfg["entropy_collapse"]:
        out.append(f"entropy collapse: {end:.2f} over the last 10 rounds (below {cfg['entropy_collapse']})")
    size = max(10, (last + 1) // 10)
    b0, b1 = avg(rows, "broken", 0, size - 1), avg(rows, "broken", last - size + 1, last)
    if last + 1 >= 2 * size and b0 is not None and b1 is not None and b1 - b0 >= cfg["broken_rise"]:
        out.append(f"rising broken share in training games: {b0:.1%} -> {b1:.1%}")
    return out

def training(folder, config):
    """The compact training record of a run folder: settings, curve blocks, key numbers and flags"""
    folder = Path(folder)
    rows, evals = read_rows(folder / "log.csv"), read_rows(folder / "eval.csv")
    info = read_json(folder / "run.json", {})
    if not rows:
        return None
    planned = info.get("config", {}).get("rounds")
    last = rows[-1]["round"]
    total = planned or last + 1
    best = max(evals, key=lambda r: r["solv_score"]) if evals else None
    return {"note": info.get("note"), "started": info.get("started"), "command": info.get("command"),
            "code": info.get("code"), "resumed": [x["at_round"] for x in info.get("resumed", [])],
            "config": info.get("config"),
            "rounds_done": last + 1, "rounds_planned": planned, "complete": bool(planned) and last >= planned - 1,
            "flags": flags(rows, evals, planned, config["flags"]),
            "scale": rows[0].get("scale"),
            "entropy": {"rounds_0_4": avg(rows, "entropy", 0, 4), "last_10": avg(rows, "entropy", last - 9, last)},
            "valid_score": {"rounds_0_4": avg(rows, "valid_score", 0, 4),
                            "last_10": avg(rows, "valid_score", last - 9, last)},
            "eval": {"first": evals[0]["solv_score"] if evals else None,
                     "rounds_10_50": avg(evals, "solv_score", 10, 50),
                     "last_200": avg(evals, "solv_score", last - 199, last),
                     "last": evals[-1]["solv_score"] if evals else None,
                     "best": {"round": best["round"], "score": best["solv_score"]} if best else None},
            "train_blocks": blocks(rows, TRAIN_KEYS, total, config["curve_blocks"]),
            "eval_blocks": blocks(evals, EVAL_KEYS, total, config["curve_blocks"])}
