import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from executor.bridge import Bridge

MAP_DIR = Path(".maps")
CATEGORIES = [
    ("not hit", "#000000"),
    ("ground", "#d9c9a3"),
    ("cliff", "#8b2e2e"),
    ("rock / mesh", "#7f7f7f"),
    ("foliage", "#3f8f3f"),
    ("building", "#e08a00"),
    ("water", "#2f6fd6"),
    ("other", "#a050c0"),
]

def category(cls, water):
    if water:
        return 6
    if cls is None:
        return 0
    if cls.startswith("Landscape"):
        return 1
    if cls == "FGCliffActor":
        return 2
    if cls == "StaticMeshActor":
        return 3
    if cls == "InstancedFoliageActor":
        return 4
    if cls.startswith("Build_"):
        return 5
    return 7

def plot(height, cats, extent, title, out, grid_step=None):
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(14, 6.5))

    terrain = plt.get_cmap("terrain").copy()
    terrain.set_bad("black")
    im = ax_h.imshow(height, origin="lower", extent=extent, cmap=terrain)
    fig.colorbar(im, ax=ax_h, label="height (m)")
    ax_h.set_title("height (black = not hit)")

    ax_c.imshow(cats, origin="lower", extent=extent, interpolation="nearest",
                cmap=ListedColormap([c for _, c in CATEGORIES]), vmin=0, vmax=len(CATEGORIES) - 1)
    ax_c.legend(handles=[Patch(color=c, label=l) for l, c in CATEGORIES],
                loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax_c.set_title("what the trace hit")

    for ax in (ax_h, ax_c):
        if grid_step:
            for v in np.arange(np.ceil(extent[0] / grid_step) * grid_step, extent[1], grid_step):
                ax.axvline(v, color="white", lw=0.6, alpha=0.7)
            for v in np.arange(np.ceil(extent[2] / grid_step) * grid_step, extent[3], grid_step):
                ax.axhline(v, color="white", lw=0.6, alpha=0.7)
        ax.set_xlabel("+y (m)")
        ax.set_ylabel("+x (m)")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)

def draw(bridge, centre, size=400.0, step=8.0, out=None):
    n = int(size / step) + 1
    x0, y0 = centre[0] - size / 2, centre[1] - size / 2
    r = bridge.sample_terrain((x0, y0), nx=n, ny=n, step=step)

    height = np.array([np.nan if z is None else z for z in r["z"]]).reshape(n, n).T
    cats = np.array([category(None if h < 0 else r["classes"][h], w)
                     for h, w in zip(r["hit"], r["water"])]).reshape(n, n).T

    extent = [y0 - step / 2, y0 + size + step / 2, x0 - step / 2, x0 + size + step / 2]
    title = (f"around ({centre[0]:.0f}, {centre[1]:.0f}), {size:.0f} m square, {step:.0f} m cells, "
             f"missed {r['missed']}, water {sum(r['water'])}")
    out = out or MAP_DIR / f"terrain_{centre[0]:.0f}_{centre[1]:.0f}.png"
    return plot(height, cats, extent, title, out)

def main():
    size = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    bridge = Bridge(timeout=60.0)
    out = draw(bridge, bridge.player()["pos"], size, step)
    print(f"saved {out}")
    if hasattr(os, "startfile"):
        os.startfile(out)

if __name__ == "__main__":
    main()