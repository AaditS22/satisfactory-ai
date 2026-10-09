import json
import math
import sys
from pathlib import Path

from executor.plan import validate_plan
from executor.build import rotate
from layout.blocks import make_block, rotate_block
from layout.export import belt, belt_class, block_belts, block_entities

CONTAINER = "Build_StorageContainerMk1_C"
FOUNDATION = "Build_Foundation_8x1_01_C"
X0, Y0 = 16.0, 7.0   # world position of block cell (0, 0)
FILL = 1200          # items per input container
PREFIX = "b"

def container(id_, x, y, fill=None):
    e = {"id": id_, "kind": "container", "class": CONTAINER,
         "pos": [float(x), float(y), 0.0], "yaw": 270}
    if fill:
        e["fill"] = fill
    return e

def foundations(entities):
    """8 m foundations covering every placed entity plus a 6 m margin."""
    xs = [e["pos"][0] for e in entities if "pos" in e]
    ys = [e["pos"][1] for e in entities if "pos" in e]
    out = []
    for fx in range(math.floor((min(xs) - 6) / 8), math.ceil((max(xs) + 6) / 8)):
        for fy in range(math.floor((min(ys) - 6) / 8), math.ceil((max(ys) + 6) / 8)):
            out.append({"id": f"f_{fx}_{fy}", "kind": "foundation", "class": FOUNDATION,
                        "pos": [8.0 * fx + 4, 8.0 * fy + 4, -0.5], "yaw": 0})
    return out

def turn(e, k, w, d):
    """Turn a placed entity by 90k degrees about cell (0, 0) of a w x d block, the
    same turn rotate_block does, so it keeps its place next to the turned block."""
    x, y = rotate(e["pos"][0] - X0, e["pos"][1] - Y0, 90 * k)
    sx, sy = {0: (0, 0), 1: (d, 0), 2: (w, d), 3: (0, w)}[k % 4]
    pos = [round(X0 + x + sx, 6), round(Y0 + y + sy, 6), e["pos"][2]]
    return {**e, "pos": pos, "yaw": (e["yaw"] + 90 * k) % 360}

def build_plan(name, recipe, n, clock, rot=0):
    b = make_block(recipe, n, clock)
    ents = []
    belts = block_belts(b, PREFIX)

    # one container per input row, 10 m left of the block, staggered 6 m so they don't touch
    first_row = b["inputs"][0]["row"]
    for j, inp in enumerate(b["inputs"]):
        cx, cy = X0 - 10, Y0 + first_row + 0.5 - 6 * j
        ents.append(container(f"in{j}", cx, cy, {"item": inp["item"], "amount": FILL}))
        belts.append(belt(belt_class(inp["rate"]), f"in{j}", "Output1", f"{PREFIX}_s{j}_0", "Input1"))

    # output container 10 m right of the block, on the output row
    out = b["output"]
    ents.append(container("out", X0 + b["w"] + 10, Y0 + out["row"] + 0.5))
    belts.append(belt(belt_class(out["rate"]), f"{PREFIX}_g{n - 1}", "Output1", "out", "Input0"))

    ents = [turn(e, rot, b["w"], b["d"]) for e in ents]
    ents = block_entities(rotate_block(b, rot), PREFIX, X0, Y0) + ents
    ents = foundations(ents) + ents
    return {"schema": "satai.factoryplan", "version": 1, "name": name, "entities": ents + belts}

def main():
    rot = 0
    if len(sys.argv) > 1:
        recipe, n, clock = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
        rot = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        name = f"block_{recipe.removeprefix('Recipe_').removesuffix('_C').lower()}_{n}"
    else:
        recipe, n, clock, name = "Recipe_IronPlateReinforced_C", 2, 1.0, "block_rip_spaced"
    if rot:
        name += f"_rot{rot}"

    plan = build_plan(name, recipe, n, clock, rot)
    problems = validate_plan(plan)
    assert not problems, problems

    path = Path("tests/plans") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    kinds = {}
    for e in plan["entities"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"wrote {path}: {kinds}")

if __name__ == "__main__":
    main()