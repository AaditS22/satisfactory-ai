import json
import math
import sys
from pathlib import Path

from executor.plan import validate_plan
from layout.blocks import make_block
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


def build_plan(name, recipe, n, clock):
    b = make_block(recipe, n, clock)
    ents = block_entities(b, PREFIX, X0, Y0)
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

    ents = foundations(ents) + ents
    return {"schema": "satai.factoryplan", "version": 1, "name": name, "entities": ents + belts}


def main():
    if len(sys.argv) > 1:
        recipe, n, clock = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
        name = f"block_{recipe.removeprefix('Recipe_').removesuffix('_C').lower()}_{n}"
    else:
        recipe, n, clock, name = "Recipe_IronPlateReinforced_C", 2, 1.0, "block_rip_spaced"

    plan = build_plan(name, recipe, n, clock)
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