import json
import math
from pathlib import Path
from solver.gamedata import recipe_by_id

DATA = Path(__file__).parent.parent / "data"
CATALOG = json.loads((DATA / "catalog.json").read_text(encoding="utf-8"))["buildings"]
SNAP = json.loads((DATA / "rules.json").read_text(encoding="utf-8"))["snap"]

def footprint(machine):
    """Calculates the footprint of a machine based on decided snapping rules"""
    sx, sy = SNAP[machine]
    (x0, y0, _), (x1, y1, _) = CATALOG[machine]["clearance_min"], CATALOG[machine]["clearance_max"]

    lower_x = math.floor(sx + x0)
    upper_x = math.ceil(sx + x1)
    width_x = upper_x - lower_x
    center_x = sx - lower_x

    lower_y = math.floor(sy + y0)
    upper_y = math.ceil(sy + y1)
    width_y = upper_y - lower_y
    center_y = sy - lower_y

    return (width_x, width_y, center_x, center_y)

def make_block(recipe_id, n, clock):
    """Creates a module for a single recipe outlining placements"""
    recipe = recipe_by_id[recipe_id]
    machine = recipe["machine"]
    mw, md, cx, cy = footprint(machine)

    width = mw * n
    num_inputs = len(recipe["inputs"])
    depth = 4 * num_inputs + 6 + md

    # centers of each machine
    centers = []
    for i in range(n):
        centers.append(((i * mw + cx), 4 * num_inputs + 1 + cy))

    # splitters to put at each input port + input rates
    ports = CATALOG[machine]["ports"]
    in_ports = sorted([p for p in ports if p["dir"] == "in"], key=lambda p: p["name"])
    splitters = []
    inputs = [] 
    for j, item in enumerate(recipe["inputs"]):
        port = in_ports[j]
        row = (num_inputs - 1 - j) * 4 + 2
        for m, (x, y) in enumerate(centers):
            splitters.append({"cell": (math.floor(x + port["pos"][0]), row),
                            "item": item, "port": port["name"],
                            "input": j, "machine": m})
        inputs.append({
            "item": item,
            "rate": recipe["inputs"][item] * n * clock,
            "row": row,
            "ends": ((-1, row), (width, row)),
        })

    # mergers to put at each output port
    out_port = [p for p in ports if p["dir"] == "out"][0]
    out_row = depth - 3
    mergers = []
    for m, (x, y) in enumerate(centers):
        mergers.append({"cell": (math.floor(x + out_port["pos"][0]), out_row), "port": out_port["name"], "machine": m})

    # output belt rate and position
    out_item = list(recipe["outputs"])[0]
    output = {
        "item": out_item,
        "rate": recipe["outputs"][out_item] * n * clock,
        "row": out_row,
        "ends": ((-1, out_row), (width, out_row)),
    }

    return {"recipe": recipe_id, "machine": machine, "n": n, "clock": clock,
        "w": width, "d": depth, "machines": centers,
        "splitters": splitters, "mergers": mergers,
        "inputs": inputs, "output": output} 