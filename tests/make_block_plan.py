import json
import math
from pathlib import Path

from layout.blocks import CATALOG, footprint, make_block
from solver.gamedata import recipe_by_id

RECIPE, N = "Recipe_IronPlateReinforced_C", 2
BELT = "Build_ConveyorBeltMk3_C"
SPLITTER, MERGER = "Build_ConveyorAttachmentSplitter_C", "Build_ConveyorAttachmentMerger_C"
CONTAINER = "Build_StorageContainerMk1_C"
X0, Y0 = 16.0, 16.0          # left edge and front (low y) edge of the machines
VARIANTS = {"tight": (0, 1), "spaced": (2, 4)}   # (gap before the machines, row pitch) in metres
FILL = 1200                  # items per input container, about 10 minutes at full speed


def build_plan(name, gap, pitch):
    recipe = recipe_by_id[RECIPE]
    machine = recipe["machine"]
    mw, md, cx, cy = footprint(machine)
    ports = CATALOG[machine]["ports"]
    in_ports = sorted([p for p in ports if p["dir"] == "in"], key=lambda p: p["name"])
    out_port = [p for p in ports if p["dir"] == "out"][0]
    items = list(recipe["inputs"])

    ents = []

    def add(id_, kind, cls, pos=None, yaw=0, **extra):
        e = {"id": id_, "kind": kind, "class": cls, **extra}
        if pos is not None:
            e["pos"], e["yaw"] = [float(v) for v in pos], yaw
        ents.append(e)

    def belt(src, src_port, dst, dst_port):
        add(f"belt_{src}_{dst}", "belt", BELT, **{"from": src, "from_port": src_port, "to": dst, "to_port": dst_port})

    centres = [X0 + m * mw + cx for m in range(N)]
    for m, x in enumerate(centres):
        add(f"asm{m}", "machine", machine, (x, Y0 + cy, 0), recipe=RECIPE)

    belt_y = [Y0 - 0.5 - gap - j * pitch for j in range(len(items))]
    out_y = Y0 + md + 0.5 + gap
    for j, item in enumerate(items):
        port = in_ports[j]
        add(f"in{j}", "container", CONTAINER, (X0 - 10, belt_y[0] - 6 * j, 0), 270, fill={"item": item, "amount": FILL})
        for m, x in enumerate(centres):
            add(f"split{j}_{m}", "attachment", SPLITTER, (x + port["pos"][0], belt_y[j], 1.0))
            belt(f"split{j}_{m}", "Output2", f"asm{m}", port["name"])
        belt(f"in{j}", "Output1", f"split{j}_0", "Input1")
        for m in range(N - 1):
            belt(f"split{j}_{m}", "Output1", f"split{j}_{m + 1}", "Input1")

    for m, x in enumerate(centres):
        add(f"merge{m}", "attachment", MERGER, (x + out_port["pos"][0], out_y, 1.0))
        belt(f"asm{m}", out_port["name"], f"merge{m}", "Input3")
    for m in range(N - 1):
        belt(f"merge{m}", "Output1", f"merge{m + 1}", "Input1")
    add("out", "container", CONTAINER, (X0 + N * mw + 10, out_y, 0), 270)
    belt(f"merge{N - 1}", "Output1", "out", "Input0")

    xs = [e["pos"][0] for e in ents if "pos" in e]
    ys = [e["pos"][1] for e in ents if "pos" in e]
    for fx in range(math.floor((min(xs) - 6) / 8), math.ceil((max(xs) + 6) / 8)):
        for fy in range(math.floor((min(ys) - 6) / 8), math.ceil((max(ys) + 6) / 8)):
            add(f"f_{fx}_{fy}", "foundation", "Build_Foundation_8x1_01_C", (8 * fx + 4, 8 * fy + 4, -0.5))

    return {"schema": "satai.factoryplan", "version": 1, "name": name, "entities": ents}


def check_tight_matches_make_block(plan):
    """The tight plan must put splitters and mergers exactly where make_block says."""
    b = make_block(RECIPE, N, 1.0)
    k = len(b["inputs"])
    want = sorted((X0 + c + 0.5, Y0 - k + r + 0.5) for (c, r), _ in b["splitters"])
    got = sorted((e["pos"][0], e["pos"][1]) for e in plan["entities"] if e["id"].startswith("split"))
    assert got == want, (got, want)
    want = sorted((X0 + c + 0.5, Y0 - k + r + 0.5) for c, r in b["mergers"])
    got = sorted((e["pos"][0], e["pos"][1]) for e in plan["entities"] if e["id"].startswith("merge"))
    assert got == want, (got, want)


def main():
    from executor.plan import validate_plan
    for name, (gap, pitch) in VARIANTS.items():
        plan = build_plan(f"block_rip_{name}", gap, pitch)
        if name == "tight":
            check_tight_matches_make_block(plan)
        problems = validate_plan(plan)
        assert not problems, problems
        path = Path("tests/plans") / f"block_rip_{name}.json"
        path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        kinds = {}
        for e in plan["entities"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        print(f"wrote {path}: {kinds}")


if __name__ == "__main__":
    main()