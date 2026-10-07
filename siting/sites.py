import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from siting.terrain import STEP, TerrainCache
from siting.ground import analyse, WATER, KIND_COLOURS, KIND_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

MAX_RANGE = 24.0 # highest minus lowest surface point allowed inside a site 
MAX_BAD = 0  # unbuildable cells allowed inside a site 
WATER_PENALTY = 2.0  # extra score for a site that is all water 
TOP_K = 5  # how many non-overlapping sites to report

MIN_SIDE = 4 
MAX_SIDE = 32 
SIDE_STEP = 2  
MAX_ASPECT = 3.0  # longest side at most 3x the shortest
ROOM_BONUS = 0.5  # score bonus per double of area above the minimum
ROOM_CAP = 2.0 

def room_bonus(area_m2, min_area_m2):
    return ROOM_BONUS * min(max(math.log2(area_m2 / min_area_m2), 0.0), ROOM_CAP)

def integral_image(a):
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    S[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return S

def window_sum(a, w, d):
    S = integral_image(a)
    return S[w:, d:] - S[:-w, d:] - S[w:, :-d] + S[:-w, :-d]

def window_max(a, w, d):
    along_x = sliding_window_view(a, w, axis=0).max(axis=-1)   
    return sliding_window_view(along_x, d, axis=1).max(axis=-1)   

def score_all(z, buildable, water, w, d):
    bad = window_sum(~buildable, w, d)
    n_good = np.maximum(w * d - bad, 1)  

    top = window_max(np.where(buildable, z, -np.inf), w, d)       
    bottom = -window_max(np.where(buildable, -z, -np.inf), w, d)  
    total = window_sum(np.where(buildable, z, 0.0), w, d)  

    fill = np.maximum(top - total / n_good, 0.0)       
    water_frac = window_sum(water & buildable, w, d) / n_good 
    height_range = top - bottom

    allowed = (bad <= MAX_BAD) & (bad < w * d) & (height_range <= MAX_RANGE)
    score = np.where(allowed, fill + WATER_PENALTY * water_frac, np.inf)
    return {"score": score, "base": top, "range": height_range, "fill": fill, "water": water_frac, "bad": bad}

def best_sites(r, w, d, k=TOP_K):
    picked = []
    for flat in np.argsort(r["score"], axis=None):  
        ix, iy = np.unravel_index(flat, r["score"].shape)
        if not np.isfinite(r["score"][ix, iy]) or len(picked) == k:
            break      
        site = {"ix": int(ix), "iy": int(iy), "w": w, "d": d,
                **{key: float(r[key][ix, iy]) for key in ("score", "base", "range", "fill", "water")}}
        if not any(_overlap(site, p) for p in picked): 
            picked.append(site)
    return picked

def candidate_sizes(min_area_m2):
    sides = range(MIN_SIDE, MAX_SIDE + 1, SIDE_STEP)
    cell_area = STEP * STEP
    return [(w, d) for w in sides for d in sides
            if w * d * cell_area >= min_area_m2 and max(w, d) / min(w, d) <= MAX_ASPECT]

def at_centres(a, w, d, px, py):
    x_lo, x_hi = (w - 1) // 2, w // 2
    y_lo, y_hi = (d - 1) // 2, d // 2
    return (a[x_lo:x_lo + px, y_lo:y_lo + py] + a[x_hi:x_hi + px, y_lo:y_lo + py]
            + a[x_lo:x_lo + px, y_hi:y_hi + py] + a[x_hi:x_hi + px, y_hi:y_hi + py]) / 4

def centres_inside(w, d, px, py, keep):
    cx = np.arange(px) + (w - 1) / 2
    cy = np.arange(py) + (d - 1) / 2
    in_x = (cx >= keep[0]) & (cx < keep[1])
    in_y = (cy >= keep[2]) & (cy < keep[3])
    return in_x[:, None] & in_y[None, :]

def find_sites(g, min_area_m2, k=TOP_K, extra_cost=None, keep=None):
    z = g.terrain.z.astype(np.float64)
    water = g.kind == WATER
    options = []
    for w, d in candidate_sizes(min_area_m2):
        r = score_all(z, g.buildable, water, w, d)
        px, py = r["score"].shape
        bonus = room_bonus(w * d * STEP * STEP, min_area_m2)
        r["score"] = r["score"] - bonus
        if extra_cost is not None:
            r["extra"] = at_centres(extra_cost, w, d, px, py)
            r["score"] = r["score"] + r["extra"]
        if keep is not None:
            r["score"] = np.where(centres_inside(w, d, px, py, keep), r["score"], np.inf)
        for s in best_sites(r, w, d, k):
            s["bonus"] = bonus
            s["extra"] = float(r["extra"][s["ix"], s["iy"]]) if extra_cost is not None else 0.0
            options.append(s)
    options.sort(key=lambda s: s["score"])
    result = []
    for s in options:
        if len(result) == k:
            break
        if not any(_overlap(s, p) for p in result):
            result.append(s)
    for s in result:
        s["centre"] = g.terrain.cell_centre(s["ix"] + (s["w"] - 1) / 2, s["iy"] + (s["d"] - 1) / 2)
    return result

def _overlap(s, p):
    return s["ix"] < p["ix"] + p["w"] and p["ix"] < s["ix"] + s["w"] and \
           s["iy"] < p["iy"] + p["d"] and p["iy"] < s["iy"] + s["d"]

def draw_sites(g, sites, title, out, links=()):
    t = g.terrain
    nx, ny = t.z.shape
    extent = [t.y0 - STEP / 2, t.y0 + ny * STEP - STEP / 2, t.x0 - STEP / 2, t.x0 + nx * STEP - STEP / 2]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(g.kind, origin="lower", extent=extent, interpolation="nearest",
              cmap=ListedColormap(KIND_COLOURS), vmin=0, vmax=len(KIND_COLOURS) - 1)
    for n, s in enumerate(sites, 1):
        rank = s.get("rank", n)
        x_lo, y_lo = t.cell_centre(s["ix"], s["iy"])
        colour = "red" if rank == 1 else "black"
        ax.add_patch(Rectangle((y_lo - STEP / 2, x_lo - STEP / 2), s["d"] * STEP, s["w"] * STEP,
                               fill=False, edgecolor=colour, lw=2))
        ax.text(s["centre"][1], s["centre"][0], str(rank), ha="center", va="center",
                fontsize=14, fontweight="bold", color=colour)
    for line in links:
        xs, ys = zip(*line)
        ax.plot(ys, xs, color="red", lw=1, alpha=0.7)
        ax.plot(ys[0], xs[0], marker="o", color="red", markersize=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(handles=[Patch(color=c, label=l) for l, c in zip(KIND_NAMES, KIND_COLOURS)],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_xlabel("+y (m)")
    ax.set_ylabel("+x (m)")
    ax.set_title(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return Path(out)

def main():
    x, y, radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    min_area = float(sys.argv[4])
    box = (x - radius, y - radius, x + radius, y + radius)
    cache = TerrainCache()
    if any(not cache.has(*c) for c in cache.chunks_for(*box)):
        print(f"not fully cached; run: python -m siting.terrain {x:.0f} {y:.0f} {radius:.0f}")
        return
    g = analyse(cache.region(*box))

    t0 = time.perf_counter()
    n_sizes = len(candidate_sizes(min_area))
    sites = find_sites(g, min_area)
    print(f"tried {n_sizes} sizes in {time.perf_counter() - t0:.2f} s")
    print(f"\nbest {len(sites)} sites with at least {min_area:.0f} m2 of floor:")
    print(f"  {'#':>2} {'centre (x, y)':>18} {'size (m)':>9} {'base z':>7} {'range':>6} {'fill':>6} "
          f"{'water':>6} {'room':>6} {'score':>6}")
    for n, s in enumerate(sites, 1):
        size = f"{s['w'] * STEP:.0f}x{s['d'] * STEP:.0f}"
        print(f"  {n:>2} {s['centre'][0]:>8.0f}, {s['centre'][1]:>7.0f} {size:>9} {s['base']:>7.1f} "
              f"{s['range']:>6.1f} {s['fill']:>6.1f} {100 * s['water']:>5.0f}% {-s['bonus']:>6.2f} {s['score']:>6.2f}")
    if not sites:
        print("  none fit")
        return

    out = draw_sites(g, sites, f"best sites (>= {min_area:.0f} m2) around ({x:.0f}, {y:.0f})",
                     Path(".maps") / f"sites_{x:.0f}_{y:.0f}.png")
    print(f"saved {out}")
    import os
    if hasattr(os, "startfile"):
        os.startfile(out)


if __name__ == "__main__":
    main()