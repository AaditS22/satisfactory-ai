import argparse

from tools import registry
from tools.baselines import baselines, eval_rows
from tools.digest import versus, label, f, cmp_text, tidy
from tools.replay import broken
from tools.stats import mean, band, bands

def stage_for(names):
    """The stage of the first name that is a run in the registry"""
    for n in names:
        s = registry.load_summary(n) if n not in ("A", "B", "specialist") else None
        if s:
            return s["stage"]
    raise ValueError(f"none of {list(names)} is a run in the registry")

def results(name, base, key="best"):
    """One model's result per task seed: a run's greedy or best layouts, baseline A or B, or the stage's specialist"""
    if name in ("A", "B"):
        return {t["seed"]: t[name] for t in base["tasks"]}
    if name == "specialist":
        name = registry.specialists().get(base["stage"])
    stored = registry.load_tasks(name) if name else None
    if stored is None:
        raise ValueError(f"{name} is not in the registry")
    if not stored["tasks"]:
        raise ValueError(f"{name} has no best_of results ({stored.get('status')}), so it cannot be compared")
    if stored["stage"] != base["stage"]:
        raise ValueError(f"{name} is a stage-{stored['stage']} run, the others are stage {base['stage']}")
    return {t["seed"]: t[key] for t in stored["tasks"]}

def side_by_side(base, x, y):
    """The A-solvable tasks both models played, with their results under 'x' and 'y'"""
    return [{**t, "x": x[t["seed"]], "y": y[t["seed"]]} for t in base["tasks"]
            if t["solvable"] and t["seed"] in x and t["seed"] in y]

def mover(r):
    """One task of a comparison with both scores and block sizes"""
    return {**label(r), "a": r["x"]["score"], "b": r["y"]["score"], "diff": r["x"]["score"] - r["y"]["score"],
            "a_kind": r["x"]["kind"], "b_kind": r["y"]["kind"], "a_blocks": r["x"]["blocks"],
            "b_blocks": r["y"]["blocks"]}

def part(rows, tie):
    """The comparison numbers of one group of tasks"""
    return {"n": len(rows), "a": mean(r["x"]["score"] for r in rows), "b": mean(r["y"]["score"] for r in rows),
            "a_broken": sum(broken(r["x"]) for r in rows), "b_broken": sum(broken(r["y"]) for r in rows),
            **versus(rows, "x", "y", tie)}

def compare_runs(a, b, key="best", movers=6):
    """Paired comparison a - b of two runs (or a run with A, B or the specialist) on the A-solvable tasks"""
    config = registry.load_config()
    tie = config["tie"]
    base = baselines(stage_for([a, b]))
    edges = config["rate_bands"].get(str(base["stage"]), config["rate_bands"]["default"])
    rows = side_by_side(base, results(a, base, key), results(b, base, key))
    by_diff = sorted(rows, key=lambda r: r["x"]["score"] - r["y"]["score"])
    big = [r for r in rows if r["big"]]
    return tidy({
        "a": a, "b": b, "layouts": key, "stage": base["stage"], "all": part(rows, tie),
        "by_rate": {n: part([r for r in rows if band(r["rate"], edges) == n], tie) for n in bands(edges)},
        "by_target": {t: part([r for r in rows if r["target"] == t], tie) for t in sorted({r["target"] for r in rows})},
        "big_block": {**part(big, tie), "tasks": [mover(r) for r in big]} if big else None,
        "a_worst": [mover(r) for r in by_diff[:movers] if r["x"]["score"] < r["y"]["score"]],
        "a_best": [mover(r) for r in by_diff[::-1][:movers] if r["x"]["score"] > r["y"]["score"]],
        "broken": {"a": [r["seed"] for r in rows if broken(r["x"])], "b": [r["seed"] for r in rows if broken(r["y"])]}})

def compare_set(runs, baseline="B", key="best"):
    """Best real score per task over a set of runs, compared with a baseline (A, B, the specialist or a run)"""
    tie = registry.load_config()["tie"]
    base = baselines(stage_for(runs))
    per_run = {r: results(r, base, key) for r in runs}
    ref = results(baseline, base, key)
    rows, wins = [], {r: 0 for r in runs}
    for t in base["tasks"]:
        if t["solvable"] and t["seed"] in ref and all(t["seed"] in per_run[r] for r in runs):
            winner = max(runs, key=lambda r: per_run[r][t["seed"]]["score"])
            wins[winner] += 1
            rows.append({**t, "x": per_run[winner][t["seed"]], "y": ref[t["seed"]], "winner": winner})
    alone = {r: mean(per_run[r][t["seed"]]["score"] for t in rows) for r in runs}
    return tidy({"runs": list(runs), "baseline": baseline, "layouts": key, "stage": base["stage"],
                          "set": part(rows, tie), "ge_baseline": sum(r["x"]["score"] >= r["y"]["score"] for r in rows),
                          "alone": alone, "wins": wins,
                          "worst": [{**mover(r), "winner": r["winner"]} for r in
                                    sorted(rows, key=lambda r: r["x"]["score"] - r["y"]["score"])[:6]]})

def lean(x, plans):
    """One layout's facts, with or without the plan itself"""
    keep = ("length", "lifts", "crossings", "floors", "penalty", "fraction", "routed", "placed")
    out = {"score": x["score"], "kind": x["kind"], "blocks": x["blocks"],
           "terms": {k: v for k, v in x["terms"].items() if k in keep}}
    if plans:
        out["plan"] = x["plan"]
    return out

def task_view(seed, runs=None, plans=True):
    """Everything known about one eval task: the task, A's and B's layouts and every run's greedy and best layouts"""
    stage = stage_for(runs) if runs else next((r["stage"] for r in eval_rows() if r["seed"] == seed), None)
    if stage is None:
        raise ValueError(f"seed {seed} is not an eval task")
    base = baselines(stage)
    t = next((t for t in base["tasks"] if t["seed"] == seed), None)
    if t is None:
        raise ValueError(f"seed {seed} is not a stage-{stage} eval task")
    runs = runs or [r["run"] for r in registry.load_index() if r["stage"] == stage]
    out = {**label(t), "stage": stage, "solvable": t["solvable"], "big_block": t["big"],
           "A": lean(t["A"], plans), "B": lean(t["B"], plans), "runs": {}}
    for r in runs:
        mine = next((x for x in (registry.load_tasks(r) or {"tasks": []})["tasks"] if x["seed"] == seed), None)
        if mine:
            out["runs"][r] = {"greedy": lean(mine["greedy"], plans), "best": lean(mine["best"], plans)}
    return tidy(out)

def group_lines(c):
    """The table rows of a two-run comparison"""
    groups = [("all tasks", c["all"])] + [("rate " + n, g) for n, g in c["by_rate"].items()] + list(c["by_target"].items())
    if c["big_block"]:
        groups.append(("big-block", c["big_block"]))
    out = ["group                    n      a      b  broken a / b  a - b"]
    for name, g in groups:
        if g["n"]:
            out.append(f"  {name:21s} {g['n']:2d}  {f(g['a'])}  {f(g['b'])}  {g['a_broken']:5d} / {g['b_broken']:<4d}  {cmp_text(g)}")
    return out

def mover_line(t):
    """One task of a comparison as text"""
    return (f"  {t['seed']} {t['target']} {t['rate']}/min {t['site']}  {f(t['a'])} vs {f(t['b'])} ({t['diff']:+.3f})  "
            f"{t['a_blocks']} vs {t['b_blocks']}" + (f"  [{t['winner']}]" if "winner" in t else ""))

def runs_text(c):
    """A two-run comparison as plain text"""
    out = [f"a = {c['a']}, b = {c['b']}  ({c['layouts']} layouts, stage {c['stage']}, A-solvable tasks)"] + group_lines(c)
    out.append(f"where both work: {cmp_text(c['all']['both_work'])}")
    if c["big_block"]:
        out += ["big-block tasks:"] + [mover_line(t) for t in c["big_block"]["tasks"]]
    out += ["a's worst tasks vs b:"] + [mover_line(t) for t in c["a_worst"]]
    out += ["a's best tasks vs b:"] + [mover_line(t) for t in c["a_best"]]
    return "\n".join(out)

def set_text(c):
    """A set comparison as plain text"""
    s = c["set"]
    out = [f"set {c['runs']} vs {c['baseline']}  ({c['layouts']} layouts, stage {c['stage']}, {s['n']} A-solvable tasks)",
           f"  best real score per task: {f(s['a'])}  ({c['baseline']} {f(s['b'])})  {cmp_text(s)}",
           f"  >= {c['baseline']} on {c['ge_baseline']} of {s['n']}; broken {s['a_broken']}"]
    out += [f"  {r:20s} alone {f(c['alone'][r])}   holds the best score on {c['wins'][r]} tasks" for r in c["runs"]]
    return "\n".join(out + ["the set's worst tasks:"] + [mover_line(t) for t in c["worst"]])

def task_text(v):
    """One task's layouts as plain text"""
    out = [f"{v['seed']} {v['target']} {v['rate']}/min {v['site']}  stage {v['stage']}"
           + ("  big-block" if v["big_block"] else "") + ("" if v["solvable"] else "  NOT A-solvable")]
    models = [("A", v["A"]), ("B", v["B"])]
    for r, x in v["runs"].items():
        models += [(f"{r} greedy", x["greedy"]), (f"{r} best", x["best"])]
    for name, x in models:
        t = x["terms"]
        nums = (f"belt {t['length']:.0f} m  lifts {t['lifts']:.0f}  crossings {t['crossings']:.0f}  floors {t['floors']:.0f}"
                if "length" in t else f"placed {t.get('placed', 0):.2f}")
        out.append(f"  {name:24s} {x['score']:7.3f} {x['kind']:7s} {nums}  blocks {x['blocks']}")
        if "plan" in x:
            out.append(f"  {'':24s} plan {x['plan']}")
    return "\n".join(out)

def list_text():
    """The registry index as plain text"""
    out = ["run                  stage  greedy (broken)  best (broken)  vs B     vs specialist   flags"]
    for r in registry.load_index():
        if r["best"] is None:
            out.append(f"{r['run']:20s} {r['stage']:5d}  {r['status']}: no best_of results" + " " * 31 + f"{len(r['flags'])}")
            continue
        out.append(f"{r['run']:20s} {r['stage']:5d}  {f(r['greedy'])} ({r['greedy_broken']:2d})      {f(r['best'])} "
                   f"({r['best_broken']:2d})     {f(r['vs_B'], True)}   {str(r['vs_specialist'] or '-'):14s}  {len(r['flags'])}")
    return "\n".join(out)

def main():
    p = argparse.ArgumentParser(description="compare runs in the registry (no replays, no best_of)")
    p.add_argument("names", nargs="*", help="two runs to compare: a b (b may be A, B or specialist)")
    p.add_argument("--greedy", action="store_true", help="compare greedy layouts instead of best-of-k")
    p.add_argument("--set", nargs="+", metavar="RUN", help="best real score per task over these runs")
    p.add_argument("--vs", default="B", help="with --set: the baseline (A, B, specialist or a run)")
    p.add_argument("--task", type=int, metavar="SEED", help="show one task's layouts (names = runs, default all)")
    p.add_argument("--list", action="store_true", help="list the runs in the registry")
    args = p.parse_args()
    key = "greedy" if args.greedy else "best"
    try:
        if args.list:
            print(list_text())
        elif args.task is not None:
            print(task_text(task_view(args.task, args.names or None)))
        elif args.set:
            print(set_text(compare_set(args.set, args.vs, key)))
        elif len(args.names) == 2:
            print(runs_text(compare_runs(args.names[0], args.names[1], key)))
        else:
            p.error("give two names, or --set, --task or --list")
    except ValueError as e:
        p.error(str(e))

if __name__ == "__main__":
    main()
