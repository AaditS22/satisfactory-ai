import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from siting.terrain import CHUNK, STEP, Terrain, TerrainCache

CLIFF_JUMP = 8.0   
ROCKS_BUILDABLE = True
PLANT_WORDS = ("Bush", "Flower", "Shroom", "Berry", "Plant")

VOID, BLOCKED, CLIFF, ROCK, WATER, LAND = range(6)
KIND_NAMES = ["void", "blocked", "cliff", "rock", "water", "land"]
KIND_COLOURS = ["#000000", "#e08a00", "#8b2e2e", "#7f7f7f", "#2f6fd6", "#d9c9a3"]


def class_kind(name):
    if name.startswith("Landscape") or name == "InstancedFoliageActor":
        return LAND
    if name == "FGCliffActor":
        return CLIFF
    if name.startswith("BP_") and any(w in name for w in PLANT_WORDS):
        return LAND
    if name == "StaticMeshActor" or name.startswith("BP_Destructible"):
        return ROCK
    return BLOCKED

@dataclass
class Ground:
    terrain: Terrain
    kind: np.ndarray 
    buildable: np.ndarray 
    slope: np.ndarray   
    room: np.ndarray    

def max_neighbour_step(z):
    step = np.zeros_like(z)
    padded = np.pad(z, 1, constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dy:padded.shape[1] - 1 + dy]
        diff = np.abs(centre - neighbour)
        step = np.fmax(step, diff)   
    step[np.isnan(z)] = np.nan
    return step

def analyse(t: Terrain) -> Ground:
    lookup = np.array([class_kind(c) for c in t.classes] + [VOID], dtype=np.int8) 
    kind = lookup[t.cls]
    kind[np.isnan(t.z)] = VOID

    kind[t.water & (kind != VOID) & (kind != CLIFF)] = WATER

    ground = (kind == LAND) | (kind == WATER)
    slope = max_neighbour_step(np.where(ground, t.z, np.nan))
    kind[ground & (slope > CLIFF_JUMP)] = CLIFF

    buildable = (kind == LAND) | (kind == WATER)
    if ROCKS_BUILDABLE:
        buildable |= kind == ROCK
    padded = np.pad(buildable, 1, constant_values=False)
    room = (ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * STEP).astype(np.float32)

    return Ground(t, kind, buildable, slope.astype(np.float32), room)

def draw_ground(g: Ground, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    t = g.terrain
    nx, ny = t.z.shape
    extent = [t.y0 - STEP / 2, t.y0 + ny * STEP - STEP / 2, t.x0 - STEP / 2, t.x0 + nx * STEP - STEP / 2]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    ax = axes[0]
    ax.imshow(g.kind, origin="lower", extent=extent, interpolation="nearest",
              cmap=ListedColormap(KIND_COLOURS), vmin=0, vmax=len(KIND_COLOURS) - 1)
    ax.legend(handles=[Patch(color=c, label=n) for n, c in zip(KIND_NAMES, KIND_COLOURS)], loc="lower left", fontsize=8)
    ax.set_title("cell kind (land + water are buildable)")

    ax = axes[1]
    slope = np.where(g.buildable, g.slope, np.nan)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#222222")
    im = ax.imshow(slope, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=CLIFF_JUMP)
    fig.colorbar(im, ax=ax, label="largest step to a neighbour (m)")
    ax.set_title("slope on buildable cells")

    ax = axes[2]
    room = np.where(g.buildable, g.room, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#222222")
    im = ax.imshow(room, origin="lower", extent=extent, cmap=cmap)
    fig.colorbar(im, ax=ax, label="distance to nearest unbuildable cell (m)")
    ax.set_title("room")

    for ax in axes:
        for v in np.arange(np.ceil(extent[0] / CHUNK) * CHUNK, extent[1], CHUNK):
            ax.axvline(v, color="white", lw=0.5, alpha=0.5)
        for v in np.arange(np.ceil(extent[2] / CHUNK) * CHUNK, extent[3], CHUNK):
            ax.axhline(v, color="white", lw=0.5, alpha=0.5)
        ax.set_xlabel("+y (m)")
        ax.set_ylabel("+x (m)")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return Path(out)

def main():
    x, y = float(sys.argv[1]), float(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    box = (x - radius, y - radius, x + radius, y + radius)

    cache = TerrainCache()
    missing = [c for c in cache.chunks_for(*box) if not cache.has(*c)]
    if missing:
        print(f"{len(missing)} chunk(s) not cached yet; run: python -m siting.terrain {x:.0f} {y:.0f} {radius:.0f}")
        return

    g = analyse(cache.region(*box))
    total = g.kind.size
    print("cells by kind:")
    for k, name in enumerate(KIND_NAMES):
        n = int((g.kind == k).sum())
        print(f"  {name:<8} {n:>7}  ({100 * n / total:4.1f}%)")
    if g.buildable.any():
        ix, iy = np.unravel_index(np.argmax(g.room), g.room.shape)
        cx, cy = g.terrain.cell_centre(ix, iy)
        print(f"most room: {g.room.max():.0f} m to the nearest obstacle, at ({cx:.0f}, {cy:.0f})")

    out = draw_ground(g, f"ground analysis around ({x:.0f}, {y:.0f})", Path(".maps") / f"ground_{x:.0f}_{y:.0f}.png")
    print(f"saved {out}")
    import os
    if hasattr(os, "startfile"):
        os.startfile(out)

if __name__ == "__main__":
    main()