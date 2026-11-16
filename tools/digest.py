import argparse
import json
import time
from pathlib import Path

from tools import registry
from tools.baselines import baselines, eval_rows, stage_of
from tools.replay import replay, broken
from tools.stats import mean, paired, band, bands

PARTS = {"belt_m": "length", "lifts": "lifts", "crossings": "crossings", "floors": "floors", "penalty": "penalty"}

def make_tasks(plans, rows, say=None):
    """Replays every greedy and best plan of a best_of file through the real scorer"""
    by_seed = {r["seed"]: r for r in rows}
    out = []
    for i, p in enumerate(plans):
        task = by_seed[p["seed"]]["task"]
        g = {**replay(task, p["greedy"]["plan"]), "saved": p["greedy"]["score"]}
        same = p["best"]["plan"] == p["greedy"]["plan"]
        b = dict(g) if same else replay(task, p["best"]["plan"])
        b["saved"] = p["best"]["score"]
        for x in (g, b):
            if x["kind"] == "error":
                x["score"] = x["saved"]
        out.append({"seed": p["seed"], "greedy": g, "best": b})
        if say and (i + 1) % 20 == 0:
            say(f"  replayed {i + 1} of {len(plans)} tasks")
    return out

def join(tasks, base, spec_tasks=None):
    """One row per task holding the run's, A's, B's and the specialist's results side by side"""
    mine = {t["seed"]: t for t in tasks}
    spec = {t["seed"]: t for t in spec_tasks} if spec_tasks else {}
    return [{**b, "greedy": mine[b["seed"]]["greedy"], "best": mine[b["seed"]]["best"],
             "spec": spec.get(b["seed"], {}).get("best")}
            for b in base["tasks"] if b["seed"] in mine]

def level(rows, key, tie):
    """Mean score, broken count and gate shares of the run's greedy or best layouts"""
    return {"mean": mean(r[key]["score"] for r in rows),
            "broken": sum(broken(r[key]) for r in rows),
            "gate_A": mean(r[key]["score"] >= r["A"]["score"] - tie for r in rows),
            "ge_B": mean(r[key]["score"] >= r["B"]["score"] for r in rows)}

def versus(rows, key, other, tie):
    """Paired comparison of the run with A, B or the specialist, on all rows and where both layouts work"""
    rows = [r for r in rows if r[other] is not None]
    both = [r for r in rows if not broken(r[key]) and not broken(r[other])]
    out = paired([r[key]["score"] for r in rows], [r[other]["score"] for r in rows], tie)
    out["both_work"] = paired([r[key]["score"] for r in both], [r[other]["score"] for r in both], tie)
    return out

def short(p):
    """A paired comparison cut down to the numbers a table needs"""
    return {k: p[k] for k in ("n", "diff", "se", "verdict")}

def group(rows, tie):
    """The numbers reported for one group of tasks (a rate band, a target, the big-block tasks)"""
    out = {"n": len(rows), "greedy": mean(r["greedy"]["score"] for r in rows),
           "best": mean(r["best"]["score"] for r in rows),
           "greedy_broken": sum(broken(r["greedy"]) for r in rows),
           "best_broken": sum(broken(r["best"]) for r in rows),
           "A": mean(r["A"]["score"] for r in rows), "B": mean(r["B"]["score"] for r in rows),
           "vs_A": short(versus(rows, "best", "A", tie)), "vs_B": short(versus(rows, "best", "B", tie))}
    if any(r["spec"] for r in rows):
        out["vs_specialist"] = short(versus(rows, "best", "spec", tie))
    return out

def parts(rows, key):
    """Mean belt metres, lifts, crossings, floors and penalty over the layouts that work"""
    ok = [r[key]["terms"] for r in rows if r[key] is not None and not broken(r[key])]
    return {"valid": len(ok), **{name: mean(t[term] for t in ok) for name, term in PARTS.items()}}

def label(r):
    """The facts that identify a task in a list"""
    return {"seed": r["seed"], "target": r["target"], "rate": r["rate"], "site": f"{r['width']}x{r['depth']}"}

def task_line(r):
    """One task with every model's best score and block sizes"""
    out = {**label(r), "greedy": r["greedy"]["score"], "best": r["best"]["score"], "A": r["A"]["score"],
           "B": r["B"]["score"], "blocks": r["best"]["blocks"], "A_blocks": r["A"]["blocks"],
           "B_blocks": r["B"]["blocks"]}
    if r["spec"]:
        out.update(specialist=r["spec"]["score"], specialist_blocks=r["spec"]["blocks"])
    return out

def mismatch(tasks):
    """Largest gap between a saved score and its replay (0 means the plans reproduce exactly)"""
    gaps = [abs(t[k]["score"] - t[k]["saved"]) for t in tasks for k in ("greedy", "best")]
    return max(gaps) if gaps else 0.0

def tidy(x):
    """Rounds every number in a summary to 5 decimals to keep the file small"""
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        return {k: tidy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tidy(v) for v in x]
    return x

def summarise(run, meta, tasks, base, spec_run=None, spec_tasks=None, config=None):
    """Builds the compact summary of a run from its per-task results"""
    config = config or registry.load_config()
    tie, big_n = config["tie"], config["big_block"]
    edges = config["rate_bands"].get(str(base["stage"]), config["rate_bands"]["default"])
    every = join(tasks, base, spec_tasks)
    rows = [r for r in every if r["solvable"]]
    rest = [r for r in every if not r["solvable"]]
    big = [r for r in rows if r["big"]]

    vs = {"A": {"greedy": versus(rows, "greedy", "A", tie), "best": versus(rows, "best", "A", tie)},
          "B": {"greedy": versus(rows, "greedy", "B", tie), "best": versus(rows, "best", "B", tie)}}
    if spec_run is None or spec_run == run:
        vs["specialist"] = {"run": spec_run, "note": "this run is the specialist" if spec_run else "no specialist set"}
    elif not spec_tasks:
        vs["specialist"] = {"run": spec_run, "note": "the specialist is not in the registry yet"}
    else:
        vs["specialist"] = {"run": spec_run, "best": versus(rows, "best", "spec", tie)}

    flags = []
    best_broken = [r for r in rows if broken(r["best"])]
    if best_broken:
        flags.append(f"{len(best_broken)} A-solvable task(s) broken in best-of-{meta['k']}: "
                     f"{[r['seed'] for r in best_broken]}")
    gap = mismatch(tasks)
    if gap > 1e-9:
        flags.append(f"replayed scores differ from the saved ones by up to {gap:.2g}: scorer or eval set changed?")
    if len(every) < len(base["tasks"]):
        flags.append(f"the plan file covers {len(every)} of the stage's {len(base['tasks'])} eval tasks")
    errors = [t["seed"] for t in tasks for k in ("greedy", "best") if t[k]["kind"] == "error"]
    if errors:
        flags.append(f"plans that could not be replayed at all: seeds {sorted(set(errors))}")

    rate_groups = {b: [r for r in rows if band(r["rate"], edges) == b] for b in bands(edges)}
    targets = sorted({r["target"] for r in rows})
    worst = sorted(rows, key=lambda r: r["best"]["score"] - r["B"]["score"])[:config["worst_tasks"]]
    model_parts = {"best": parts(rows, "best"), "greedy": parts(rows, "greedy"), "A": parts(rows, "A"),
                   "B": parts(rows, "B")}
    if spec_tasks and spec_run != run:
        model_parts["specialist"] = parts(rows, "spec")

    return tidy({"run": run, "stage": base["stage"], "source": meta, "tasks": len(every), "solvable": len(rows),
            "flags": flags,
            "greedy": level(rows, "greedy", tie), "best": level(rows, "best", tie),
            "A": mean(r["A"]["score"] for r in rows), "B": mean(r["B"]["score"] for r in rows),
            "vs": vs,
            "by_rate": {b: group(g, tie) for b, g in rate_groups.items() if g},
            "by_target": {t: group([r for r in rows if r["target"] == t], tie) for t in targets},
            "big_block": {"rule": f"A's biggest block is {big_n}+", **group(big, tie),
                          "used_big": sum(max(r["best"]["blocks"]) >= big_n for r in big),
                          "tasks": [task_line(r) for r in big]} if big else None,
            "parts": model_parts,
            "broken": {k: [{**label(r), "kind": r[k]["kind"], "score": r[k]["score"], "blocks": r[k]["blocks"]}
                           for r in rows if broken(r[k])] for k in ("greedy", "best")},
            "worst_vs_B": [task_line(r) for r in worst],
            "unsolvable": {"n": len(rest), "best_works": sum(not broken(r["best"]) for r in rest)},
            "replay_gap": gap})

def digest(run, plans, meta):
    """Replays a best_of plan file and stores the run's per-task results and summary in the registry"""
    rows = eval_rows()
    stage = stage_of([p["seed"] for p in plans], rows)
    base = baselines(stage)
    tasks = make_tasks(plans, [r for r in rows if r["stage"] == stage], say=print)
    stored = {"run": run, "stage": stage, "source": meta, "tasks": tasks}
    summary = resummarise(stored, base)
    registry.save_run(summary, stored)
    return summary

def resummarise(stored, base=None):
    """Builds a run's summary again from its stored per-task results (no replays)"""
    base = base or baselines(stored["stage"])
    spec_run = registry.specialists().get(stored["stage"])
    spec = registry.load_tasks(spec_run) if spec_run and spec_run != stored["run"] else None
    return summarise(stored["run"], stored["source"], stored["tasks"], base, spec_run, spec and spec["tasks"])

def refresh():
    """Rebuilds every summary, e.g. after the specialists file or the config changed"""
    for run in registry.all_runs():
        registry.save_run(resummarise(registry.load_tasks(run)))
        print(f"refreshed {run}")

def f(x, sign=False):
    """A number as text with 3 decimals, or '-' if there is none"""
    return "-" if x is None else f"{x:+.3f}" if sign else f"{x:.3f}"

def cmp_text(p):
    """A paired comparison as one piece of text"""
    if p["diff"] is None:
        return "-"
    se = f"{p['n_se']:.1f} SE" if p.get("n_se") is not None else f"SE {f(p['se'])}"
    count = f"; {p['better']} / {p['worse']}" if "better" in p else ""
    return f"{p['diff']:+.4f} ({se}{count}) {p['verdict']}"

def report(s):
    """The summary as plain text"""
    k = s["source"]["k"]
    out = [f"{s['run']}  stage {s['stage']}  {s['solvable']} A-solvable tasks of {s['tasks']}  "
           f"(A {f(s['A'])}, B {f(s['B'])})"]
    out += [f"  FLAG: {x}" for x in s["flags"]]
    for name, key in (("greedy", "greedy"), (f"best-of-{k}", "best")):
        v = s[key]
        out.append(f"{name:11s} {f(v['mean'])}  broken {v['broken']}  >= A-tie {v['gate_A']:.0%}  >= B {v['ge_B']:.0%}"
                   f"  vs A {cmp_text(s['vs']['A'][key])}  vs B {cmp_text(s['vs']['B'][key])}")
    spec = s["vs"]["specialist"]
    if "best" in spec:
        out.append(f"vs specialist {spec['run']} (best-of-{k}): {cmp_text(spec['best'])}"
                   f"   where both work: {cmp_text(spec['best']['both_work'])}")
    else:
        out.append(f"vs specialist: {spec['note']}" + (f" ({spec['run']})" if spec["run"] else ""))
    groups = [("rate " + b, g) for b, g in s["by_rate"].items()] + list(s["by_target"].items())
    if s["big_block"]:
        groups.append((f"big-block ({s['big_block']['used_big']} of {s['big_block']['n']} use one)", s["big_block"]))
    out.append("group                             n  greedy   best  broken  vs A     vs B     vs specialist")
    for name, g in groups:
        sp = g.get("vs_specialist", {}).get("diff")
        out.append(f"  {name:30s} {g['n']:2d}  {f(g['greedy'])}  {f(g['best'])}  {g['best_broken']:6d}  "
                   f"{f(g['vs_A']['diff'], True)}   {f(g['vs_B']['diff'], True)}   {f(sp, True)}")
    out.append("parts (working layouts)   belt m  lifts  crossings  floors  penalty")
    for name, p in s["parts"].items():
        if p["valid"]:
            out.append(f"  {name:12s} ({p['valid']:3d})      {p['belt_m']:6.0f}  {p['lifts']:5.2f}  {p['crossings']:9.2f}  "
                       f"{p['floors']:6.2f}  {f(p['penalty'])}")
    for key in ("greedy", "best"):
        for b in s["broken"][key]:
            out.append(f"  broken {key}: {b['seed']} {b['target']} {b['rate']}/min {b['site']} {b['kind']} "
                       f"{b['score']:.3f} blocks {b['blocks']}")
    out.append("worst tasks vs B:")
    for t in s["worst_vs_B"]:
        out.append(f"  {t['seed']} {t['target']} {t['rate']}/min {t['site']}  {f(t['best'])} vs B {f(t['B'])} "
                   f"({t['best'] - t['B']:+.3f})  blocks {t['blocks']}  B {t['B_blocks']}")
    return "\n".join(out)

def main():
    p = argparse.ArgumentParser(description="turn a best_of plan file into a permanent registry entry")
    p.add_argument("plans", nargs="?", help="the file written by layout.best_of --save")
    p.add_argument("--run", help="name of the registry entry, e.g. stage3_i")
    p.add_argument("--k", type=int, default=64, help="the --k best_of was run with")
    p.add_argument("--seed", type=int, default=0, help="the --seed best_of was run with")
    p.add_argument("--checkpoint", default="last.pt", help="which checkpoint the plans came from")
    p.add_argument("--note", default="", help="anything worth remembering about this entry")
    p.add_argument("--force", action="store_true", help="replace an entry that already exists")
    p.add_argument("--refresh", action="store_true", help="rebuild every summary from the stored per-task results")
    p.add_argument("--show", metavar="RUN", help="print a stored summary")
    args = p.parse_args()

    if args.refresh:
        return refresh()
    if args.show:
        s = registry.load_summary(args.show)
        return print(report(s) if s else f"{args.show} is not in the registry")
    if not args.plans or not args.run:
        p.error("give a plan file and --run (or use --refresh / --show)")
    registry.check_name(args.run)
    if registry.load_tasks(args.run) and not args.force:
        p.error(f"{args.run} is already in the registry: add --force to replace it")

    plans = json.loads(Path(args.plans).read_text())
    meta = {"plans": Path(args.plans).name, "k": args.k, "seed": args.seed, "checkpoint": args.checkpoint,
            "note": args.note, "digested": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    summary = digest(args.run, plans, meta)
    print(f"stored runs/registry/{args.run}/ in {time.time() - t0:.0f} s\n")
    print(report(summary))

if __name__ == "__main__":
    main()
