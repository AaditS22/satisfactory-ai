import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

from solver.gamedata import recipe_by_id, items
from layout.blocks import make_block, footprint, rot_point

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948",
           "#9c755f", "#ff9da7", "#7f7f7f", "#003f5c", "#ffa600", "#a05195", "#2ca02c",
           "#8c564b", "#17becf", "#bcbd22", "#d45087", "#665191", "#bab0ac"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BELT = "#1b1b1b"
FOUNDATION = 8

def recipe_colours(blocks):
    """{recipe id: (colour, letter)}, in the order recipes were first placed."""
    colours = {}
    for b in blocks:
        r = b["block"]["recipe"]
        if r not in colours:
            i = len(colours)
            colours[r] = (PALETTE[i % len(PALETTE)], LETTERS[i % len(LETTERS)])
    return colours

def recipe_label(rid):
    """Short readable recipe name: 'Alternate: Steel Rod' -> 'Steel Rod (alt)'."""
    name = recipe_by_id[rid]["name"]
    if name.startswith("Alternate: "):
        return name[len("Alternate: "):] + " (alt)"
    return name

def turn(points, k, w, d):
    """Points of an unrotated w x d block, after the block is turned k times (same as rotate_block)."""
    out = []
    for p in points:
        x, y = p
        ww, dd = w, d
        for _ in range(k % 4):
            x, y = rot_point(x, y, ww, dd)
            ww, dd = dd, ww
        out.append((x, y))
    return out

def block_shapes(blk):
    """Machine boxes, belt lines and flow arrows of a placed block, in its own (rotated) coordinates."""
    base = make_block(blk["recipe"], blk["n"], blk["clock"])
    w, d, k = base["w"], base["d"], blk["rot"]
    mw, md, cx, cy = footprint(base["machine"])

    boxes = [(x - cx, y - cy, x - cx + mw, y - cy + md) for x, y in base["machines"]]
    belts, arrows = [], []

    for j, inp in enumerate(base["inputs"]):
        row = inp["row"] + 0.5
        cells = [s["cell"][0] for s in base["splitters"] if s["input"] == j]
        belts.append([(-0.5, row), (max(cells) + 0.5, row)])
        arrows.append(((-0.5, row), (0.5, row)))
        for s in base["splitters"]:
            if s["input"] == j:
                c = s["cell"][0] + 0.5
                belts.append([(c, row), (c, boxes[s["machine"]][1])])

    out_row = base["output"]["row"] + 0.5
    cells = [m["cell"][0] for m in base["mergers"]]
    belts.append([(min(cells) + 0.5, out_row), (w + 0.5, out_row)])
    arrows.append(((w - 0.5, out_row), (w + 0.5, out_row)))
    for m in base["mergers"]:
        c = m["cell"][0] + 0.5
        belts.append([(c, boxes[m["machine"]][3]), (c, out_row)])

    machines = []
    for x0, y0, x1, y1 in boxes:
        (ax, ay), (bx, by) = turn([(x0, y0), (x1, y1)], k, w, d)
        machines.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
    belts = [turn(line, k, w, d) for line in belts]
    arrows = [tuple(turn(a, k, w, d)) for a in arrows]
    return {"machines": machines, "belts": belts, "arrows": arrows}

def _inward(x, y, X, Y):
    """Text offset (points) that points from an edge cell into the site, so labels stay inside."""
    dx = -7 if x == X - 1 else 7
    dy = -6 if y == Y - 1 else 5
    return dx, dy

def _draw_floor(ax, env, f, colours):
    F, X, Y = env.world.owner.shape
    ax.set_xticks(range(0, X + 1, FOUNDATION))
    ax.set_yticks(range(0, Y + 1, FOUNDATION))
    ax.grid(True, color="#d9d9d9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.add_patch(Rectangle((0, 0), X, Y, fill=False, ec="#333333", lw=1.6, zorder=1))

    for i, b in enumerate(env.world.blocks):
        if b["floor"] != f:
            continue
        blk, x0, y0 = b["block"], b["x"], b["y"]
        col, letter = colours[blk["recipe"]]
        ax.add_patch(Rectangle((x0, y0), blk["w"], blk["d"], fc=col, alpha=0.28, ec="none", zorder=2))
        ax.add_patch(Rectangle((x0, y0), blk["w"], blk["d"], fill=False, ec=col, lw=1.8, zorder=5))
        shapes = block_shapes(blk)
        for mx0, my0, mx1, my1 in shapes["machines"]:
            ax.add_patch(Rectangle((x0 + mx0, y0 + my0), mx1 - mx0, my1 - my0,
                                   fc=col, alpha=0.85, ec="white", lw=0.6, zorder=3))
        for line in shapes["belts"]:
            xs = [x0 + p[0] for p in line]
            ys = [y0 + p[1] for p in line]
            ax.plot(xs, ys, color=BELT, lw=2.0, solid_capstyle="round", zorder=4)
        for (sx, sy), (ex, ey) in shapes["arrows"]:
            ax.add_patch(FancyArrowPatch((x0 + sx, y0 + sy), (x0 + ex, y0 + ey), arrowstyle="-|>",
                                         mutation_scale=14, color=BELT, lw=0, zorder=6))
        for cx, cy in b["ends"]:
            ax.plot(cx + 0.5, cy + 0.5, "o", ms=3.5, mfc="white", mec=BELT, mew=1.0, zorder=7)
        ax.text(x0 + blk["w"] / 2, y0 + blk["d"] / 2, f"{letter} ×{blk['n']}", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5, alpha=0.95))
        ax.text(x0 + 0.4, y0 + 0.4, f"#{i}", ha="left", va="bottom", fontsize=6, color="#333333", zorder=8)

    if f == 0:
        for item, (ex, ey) in env.entries.items():
            ax.plot(ex + 0.5, ey + 0.5, "^", color="#2e8b3e", ms=10, mec="black", mew=0.6, zorder=9)
            ax.annotate(items[item]["name"], (ex + 0.5, ey + 0.5), xytext=_inward(ex, ey, X, Y),
                        textcoords="offset points", ha="right" if ex == X - 1 else "left",
                        va="top" if ey == Y - 1 else "bottom",
                        fontsize=7.5, color="#1d5c28", fontweight="bold", zorder=9)
        ox, oy = env.exit
        ax.plot(ox + 0.5, oy + 0.5, "s", color="#d62728", ms=9, mec="black", mew=0.6, zorder=9)
        ax.annotate("OUT", (ox + 0.5, oy + 0.5), xytext=_inward(ox, oy, X, Y), textcoords="offset points",
                    ha="right" if ox == X - 1 else "left", va="top" if oy == Y - 1 else "bottom", fontsize=7.5, color="#a01c1c", fontweight="bold", zorder=9)

    ax.set_xlim(-2, X + 2)
    ax.set_ylim(-2, Y + 2)
    ax.set_aspect("equal")
    ax.set_title(f"floor {f}", fontsize=11)
    ax.tick_params(labelsize=7)

def draw(env, path, title=None):
    """Save a PNG of env's current layout (any time: mid-episode, finished or failed)."""
    F, X, Y = env.world.owner.shape
    colours = recipe_colours(env.world.blocks)

    aspect = X / Y
    if aspect >= 1.3:
        nrows, ncols = F, 1
        pw, ph = 10.0, 10.0 / aspect
    else:
        nrows, ncols = 1, F
        pw, ph = 6.5 * aspect, 6.5
    width = max(pw * ncols + 0.6, 10.0)
    legend_cols = max(1, int(width // 3.4))
    n_items = len(colours) + 1
    legend_rows = (n_items + legend_cols - 1) // legend_cols
    height = ph * nrows + 1.0 + 0.28 * legend_rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    for f in range(F):
        _draw_floor(axes.flat[f], env, f, colours)

    totals = {}
    for b in env.world.blocks:
        totals[b["block"]["recipe"]] = totals.get(b["block"]["recipe"], 0) + b["block"]["n"]
    handles = [Patch(fc=col, ec=col, label=f"{letter}: {recipe_label(r)} ({totals[r]})")
               for r, (col, letter) in colours.items()]
    handles.append(Line2D([], [], color=BELT, lw=2.0, label="belt (arrow = flow direction)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(legend_cols, len(handles)), fontsize=8, frameon=False)

    if title is None:
        graph = env.graph
        state = "finished" if env.done and not env.failed else ("FAILED: no room" if env.failed else "in progress")
        left = sum(e["remaining"] for e in env.queue)
        unplaced = f", {left} machine{'s' if left != 1 else ''} unplaced" if left else ""
        title = (f"{items[graph['target']]['name']} at {graph['rate']}/min\n"
                 f"site {X}×{Y} m, {F} floors  ·  {len(env.world.blocks)} blocks, {state}{unplaced}")
    fig.suptitle(title, fontsize=11)
    bottom = (0.15 + 0.28 * legend_rows) / height
    top = 1 - 0.75 / height
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(path, dpi=130)
    plt.close(fig)

def random_episode(seed):
    """Play one random-agent episode on a random task; returns the finished env."""
    from layout.env import LayoutEnv
    from layout.tasks import random_task

    rng = np.random.default_rng(seed)
    env = LayoutEnv()
    env.reset(random_task(rng))
    while not env.done:
        ns = np.flatnonzero(env.n_mask())
        n = int(ns[rng.integers(len(ns))])
        spots = np.argwhere(env.pos_mask(n))
        rot, f, x, y = (int(v) for v in spots[rng.integers(len(spots))])
        env.step(n, rot, f, x, y)
    return env

def main():
    parser = argparse.ArgumentParser(description="Play one random-agent episode and save a picture of it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="layout.png")
    args = parser.parse_args()
    env = random_episode(args.seed)
    draw(env, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()