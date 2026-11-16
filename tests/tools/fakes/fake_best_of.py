"""A stand-in for layout.best_of used only by the supervisor tests: 'plays' A's plans as greedy and B's as best"""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stage", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--save")
    args = p.parse_args()
    rows = [r for r in json.loads(Path("data/eval_set.json").read_text()) if r["stage"] == args.stage]
    out = [{"seed": r["seed"], "greedy": {"plan": r["A"]["plan"], "score": r["A"]["score"]},
            "best": {"plan": r["B"]["plan"], "score": r["B"]["score"]}} for r in rows]
    Path(args.save).write_text(json.dumps(out))

if __name__ == "__main__":
    main()
