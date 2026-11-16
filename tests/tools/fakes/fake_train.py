"""A stand-in for layout.train used only by the supervisor tests: writes log rows fast and starts real spawn workers"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

COLS = ["round", "score", "failed", "invalid", "valid_score", "entropy", "approx_kl", "value_loss", "policy_loss",
        "clip_frac", "push", "collect_s", "update_s", "flat", "scale"]

def idle():
    """A worker that never notices its parent is gone"""
    while True:
        time.sleep(0.2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--stage", type=int)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--resume", action="store_true")
    args, _ = p.parse_known_args()
    folder = Path(f"runs/stage{args.stage}/{args.run}")
    folder.mkdir(parents=True, exist_ok=True)
    mode = args.note if not args.resume else ""
    start = 0
    if args.resume:
        start = len((folder / "log.csv").read_text().splitlines()) - 1
    else:
        (folder / "run.json").write_text(json.dumps({"note": args.note, "started": "now", "command": " ".join(sys.argv),
                                                     "code": "fake", "resumed": [],
                                                     "config": {"stage": args.stage, "rounds": args.rounds,
                                                                "seed": args.seed, "workers": 2}}))
    workers = [mp.get_context("spawn").Process(target=idle) for _ in range(2)]
    for w in workers:
        w.start()
    step = float(os.environ.get("FAKE_ROUND_S", "0.02"))
    for name, cols in (("log.csv", COLS), ("eval.csv", ["round", "solv_score", "gate_A", "failed", "invalid"])):
        if not args.resume:
            with open(folder / name, "w", newline="") as fh:
                csv.writer(fh).writerow(cols)
    for r in range(start, args.rounds):
        x = r / args.rounds
        entropy, valid = 1.3 - 0.7 * x, 0.59 + 0.02 * x
        if "drift" in mode and r >= 5:
            entropy, valid = 1.9, 0.55
        if "nan" in mode and r >= 8:
            entropy = float("nan")
        with open(folder / "log.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([r, valid - 0.1, 0.03, 0.03, valid, entropy, 0.05, 0.1, 0.01, 0.2, 0.7, 3.0, 3.0,
                                     0.01, 0.061])
        if r % 10 == 0 or r == args.rounds - 1:
            with open(folder / "eval.csv", "a", newline="") as fh:
                csv.writer(fh).writerow([r, 0.5, 0.4, 0.02, 0.03])
            (folder / "last.pt").write_text("fake")
        if "hang" in mode and r == 12:
            time.sleep(10_000)
        if "orphan" in mode and r == 12:
            os._exit(1)
        if "stop_half" in mode and r == args.rounds // 2:
            for w in workers:
                w.kill()
            sys.exit(3)
        time.sleep(step)
    for w in workers:
        w.kill()

if __name__ == "__main__":
    main()
