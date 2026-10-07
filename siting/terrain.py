import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from executor.bridge import Bridge

STEP = 8.0   
CELLS = 64 
CHUNK = STEP * CELLS    
HOVER = 50.0   
SETTLE = 0.5     
STREAM_TIMEOUT = 20.0
PAUSE_AFTER = 1.0 

@dataclass
class Terrain:
    x0: float 
    y0: float
    z: np.ndarray     
    cls: np.ndarray      
    water: np.ndarray 
    classes: list

    def cell_centre(self, ix, iy):
        return self.x0 + ix * STEP, self.y0 + iy * STEP

class TerrainCache:
    def __init__(self, save_name="SatAI_Test", root=".terrain"):
        self.dir = Path(root) / save_name
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def chunks_for(x_min, y_min, x_max, y_max):
        i0, i1 = math.floor(x_min / CHUNK), math.floor(x_max / CHUNK)
        j0, j1 = math.floor(y_min / CHUNK), math.floor(y_max / CHUNK)
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def path(self, i, j):
        return self.dir / f"chunk_{i}_{j}.npz"

    def has(self, i, j):
        return self.path(i, j).exists()

    def ensure(self, bridge, x_min, y_min, x_max, y_max, z_guess=0.0, log=print, return_home=True):
        todo = [c for c in self.chunks_for(x_min, y_min, x_max, y_max) if not self.has(*c)]
        if not todo:
            return 0
        todo.sort(key=lambda c: (c[0], c[1] if c[0] % 2 == 0 else -c[1]))

        home = bridge.player()["pos"] if return_home else None
        log(f"sampling {len(todo)} chunk(s); you'll be teleported and frozen" + (", then brought home" if return_home else ""))
        try:
            for n, (i, j) in enumerate(todo, 1):
                t = time.monotonic()
                cx, cy = (i + 0.5) * CHUNK, (j + 0.5) * CHUNK
                bridge.teleport((cx, cy, z_guess + HOVER), freeze=True)
                loaded = self._wait_for_streaming(bridge)
                self._sample_chunk(bridge, i, j)
                time.sleep(PAUSE_AFTER) 
                log(f"  chunk ({i}, {j}) {n}/{len(todo)}: {time.monotonic() - t:.1f} s"
                    + ("" if loaded else "  (WARNING: streaming did not report complete)"))
        finally:
            if return_home:
                bridge.teleport((home[0], home[1], home[2] + 1), freeze=False)
        return len(todo)

    @staticmethod
    def _wait_for_streaming(bridge):
        time.sleep(SETTLE)
        start = time.monotonic()
        streak = 0
        while time.monotonic() - start < STREAM_TIMEOUT:
            streak = streak + 1 if bridge.streaming_complete() else 0
            if streak >= 2:  
                return True
            time.sleep(0.25)
        return False

    def _sample_chunk(self, bridge, i, j):
        x0, y0 = i * CHUNK + STEP / 2, j * CHUNK + STEP / 2
        r = bridge.sample_terrain((x0, y0), nx=CELLS, ny=CELLS, step=STEP)
        z = np.array([np.nan if v is None else v for v in r["z"]], dtype=np.float32).reshape(CELLS, CELLS).T
        cls = np.array(r["hit"], dtype=np.int16).reshape(CELLS, CELLS).T
        water = np.array(r["water"], dtype=bool).reshape(CELLS, CELLS).T
        np.savez_compressed(self.path(i, j), z=z, cls=cls, water=water, classes=np.array(r["classes"], dtype=str))

    def region(self, x_min, y_min, x_max, y_max):
        chunks = self.chunks_for(x_min, y_min, x_max, y_max)
        i0, j0 = min(c[0] for c in chunks), min(c[1] for c in chunks)
        ni = max(c[0] for c in chunks) - i0 + 1
        nj = max(c[1] for c in chunks) - j0 + 1
        z = np.full((ni * CELLS, nj * CELLS), np.nan, dtype=np.float32)
        cls = np.full(z.shape, -1, dtype=np.int16)
        water = np.zeros(z.shape, dtype=bool)
        classes, class_index = [], {}

        for i, j in chunks:
            if not self.has(i, j):
                continue
            d = np.load(self.path(i, j))
            remap = np.array([class_index.setdefault(c, len(class_index)) for c in d["classes"]] + [-1], dtype=np.int16)
            for c in d["classes"]:
                if c not in classes:
                    classes.append(c)
            sl = np.s_[(i - i0) * CELLS:(i - i0 + 1) * CELLS, (j - j0) * CELLS:(j - j0 + 1) * CELLS]
            z[sl], water[sl] = d["z"], d["water"]
            cls[sl] = remap[d["cls"]] 

        return Terrain(i0 * CHUNK + STEP / 2, j0 * CHUNK + STEP / 2, z, cls, water, classes)


def draw_terrain(t, title, out):
    from executor.terrain_map import category, plot 
    cats = np.vectorize(lambda c, w: category(None if c < 0 else t.classes[c], w))(t.cls, t.water)
    nx, ny = t.z.shape
    extent = [t.y0 - STEP / 2, t.y0 + ny * STEP - STEP / 2, t.x0 - STEP / 2, t.x0 + nx * STEP - STEP / 2]
    return plot(t.z, cats, extent, title, out, grid_step=CHUNK)


def main():
    x, y = float(sys.argv[1]), float(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    box = (x - radius, y - radius, x + radius, y + radius)

    bridge = Bridge(timeout=60.0)
    cache = TerrainCache()
    nodes = bridge.resource_nodes()
    z_guess = min(nodes, key=lambda n: math.hypot(n["pos"][0] - x, n["pos"][1] - y))["pos"][2]

    t0 = time.monotonic()
    sampled = cache.ensure(bridge, *box, z_guess=z_guess)
    print(f"{sampled} chunk(s) sampled in {time.monotonic() - t0:.1f} s"
          + ("" if sampled else " (everything was already cached)"))

    t = cache.region(*box)
    title = (f"cached terrain around ({x:.0f}, {y:.0f}): {t.z.shape[0]} x {t.z.shape[1]} cells, "
             f"missed {int(np.isnan(t.z).sum())}, water {int(t.water.sum())}")
    out = draw_terrain(t, title, Path(".maps") / f"cache_{x:.0f}_{y:.0f}.png")
    print(f"saved {out}")
    import os
    if hasattr(os, "startfile"):
        os.startfile(out)


if __name__ == "__main__":
    main()