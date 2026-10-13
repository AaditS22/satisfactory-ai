from solver.graph import pick_belt
import json
import math

from layout.blocks import CATALOG
from layout.connect import chains
from layout.flow import end_owner

import argparse
from pathlib import Path
from executor.plan import validate_plan
from layout.api import load, evaluate
from layout.view import draw

SPLITTER = "Build_ConveyorAttachmentSplitter_C"
MERGER = "Build_ConveyorAttachmentMerger_C"

FLOOR_H = 16.0
BELT_Z = 1.0
CONTAINER = "Build_StorageContainerMk1_C"
FOUNDATION = "Build_Foundation_8x1_01_C"
FILL = 1200
CONTAINER_GAP = 8.0
MAX_RUN = 48
MIN_BELT = 1.0
YAW_X = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}
YAW_Y = {(0, 1): 0, (-1, 0): 90, (0, -1): 180, (1, 0): 270}
INWARD = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def block_entities(block, prefix, x0, y0):
    """Defines entities following the set schema for building"""
    yaw = block["rot"] * 90

    entities = []
    for m, (x, y) in enumerate(block["machines"]):
        entities.append({"id": f"{prefix}_m{m}", "kind": "machine", "class": block["machine"],
                         "pos": [x0 + x, y0 + y, 0.0], "yaw": yaw,
                         "recipe": block["recipe"], "clock": block["clock"]})

    for s in block["splitters"]:
        c, r = s["cell"]
        entities.append({"id": f"{prefix}_s{s['input']}_{s['machine']}", 
                         "kind": "attachment", "class": SPLITTER, "yaw": yaw,
                         "pos" : [x0 + c + 0.5, y0 + r + 0.5, 1.0]})

    for m in block["mergers"]:
        c, r = m["cell"]
        entities.append({"id": f"{prefix}_g{m['machine']}", 
                        "kind": "attachment", "class": MERGER, "yaw": yaw,
                        "pos" : [x0 + c + 0.5, y0 + r + 0.5, 1.0]})    
    return entities

def belt_class(rate):
    """Finds the class for a belt based on its tier"""
    tier, _ = pick_belt(rate)
    return f"Build_ConveyorBeltMk{tier}_C"


def belt(cls, from_id, from_port, to_id, to_port):
    """Creates a single belt entity"""
    return {"id": f"belt_{from_id}_{to_id}", "kind": "belt", "class": cls,
            "from": from_id, "from_port": from_port, "to": to_id, "to_port": to_port}


def block_belts(block, prefix):
    n = block["n"]
    entities = []

    # belts from splitter to amchine
    for s in block["splitters"]:
        s_id = f"{prefix}_s{s['input']}_{s['machine']}"
        m_id = f"{prefix}_m{s['machine']}"
        input_cls = belt_class(block["inputs"][s["input"]]["rate"])
        entities.append(belt(input_cls, s_id, "Output2", m_id, s["port"]))

    # belts from each splitter to the next
    for j, row in enumerate(block["inputs"]):
        input_cls = belt_class(row["rate"])
        for m in range(n - 1):
            from_id = f"{prefix}_s{j}_{m}"
            to_id = f"{prefix}_s{j}_{m + 1}"
            entities.append(belt(input_cls, from_id, "Output1",
                                 to_id, "Input1"))

    # belts from machines to mergers
    output_cls = belt_class(block["output"]["rate"])
    for g in block["mergers"]:
        m = g["machine"]
        from_id = f"{prefix}_m{m}"
        to_id = f"{prefix}_g{m}"
        entities.append(belt(output_cls, from_id, g["port"], to_id, "Input3"))

    # belts from mergers to the next merger    
    for m in range(n - 1):
        entities.append(belt(output_cls, f"{prefix}_g{m}", "Output1", f"{prefix}_g{m + 1}", "Input1"))

    return entities

def turn_vec(v, yaw):
    """Rotate a 2D vector by yaw degrees"""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (round(v[0] * c - v[1] * s), round(v[0] * s + v[1] * c))

def port_facing(cls, yaw):
    return {p["name"]: turn_vec(p["facing"][:2], yaw) for p in CATALOG[cls]["ports"]}

def port_with(cls, yaw, facing, direction):
    for p in CATALOG[cls]["ports"]:
        if p["dir"] == direction and turn_vec(p["facing"][:2], yaw) == facing:
            return p["name"]
    raise ValueError(f"{cls} at yaw {yaw} has no {direction} port facing {facing}")

def port_pos(entity, port):
    """Finds the position of a port of a placed entity."""
    p = next(q for q in CATALOG[entity["class"]]["ports"] if q["name"] == port)
    c, s = math.cos(math.radians(entity["yaw"])), math.sin(math.radians(entity["yaw"]))
    x, y, z = p["pos"]
    return (round(entity["pos"][0] + x * c - y * s, 6), round(entity["pos"][1] + x * s + y * c, 6),
            entity["pos"][2] + z)

def cell_pos(f, x, y, z=BELT_Z):
    return [x + 0.5, y + 0.5, f * FLOOR_H + z]

def step(a, b):
    """Direction of one same-floor path step a -> b"""
    return (b[1] - a[1], b[2] - a[2])

def block_side(env, cell, bid):
    """Direction from an end cell into the block that owns it"""
    f, x, y = cell
    for d in YAW_X:
        nx, ny = x + d[0], y + d[1]
        if 0 <= nx < env.world.owner.shape[1] and 0 <= ny < env.world.owner.shape[2] \
                and env.world.owner[f, nx, ny] == bid:
            return d
    raise ValueError(f"end cell {cell} is not next to block {bid}")

def floor_foundations(entities):
    """Places 8m foundations under every entity"""
    by_floor = {}
    for e in entities:
        if "pos" in e:
            by_floor.setdefault(int(round(e["pos"][2] // FLOOR_H)), []).append(e)
    out = []
    for f, ents in sorted(by_floor.items()):
        xs = [e["pos"][0] for e in ents]
        ys = [e["pos"][1] for e in ents]
        for fx in range(math.floor((min(xs) - 6) / 8), math.ceil((max(xs) + 6) / 8)):
            for fy in range(math.floor((min(ys) - 6) / 8), math.ceil((max(ys) + 6) / 8)):
                out.append({"id": f"f{f}_{fx}_{fy}", "kind": "foundation", "class": FOUNDATION,
                            "pos": [8.0 * fx + 4, 8.0 * fy + 4, f * FLOOR_H - 0.5], "yaw": 0})
    return out

def layout_plan(env, routes, name):
    """Creates a FactoryPlan for a full finished episode"""
    if not env.done or env.failed:
        raise ValueError("episode not finished")
    if routes["failed"]:
        raise ValueError(f"{len(routes['failed'])} link(s) not routed")
    for path in routes["paths"].values():
        if any(a[0] != b[0] for a, b in zip(path, path[1:])):
            raise ValueError("layout uses lifts; the executor can't build lifts yet (Phase 6)")

    ents, belts = [], []
    by_id = {}

    def add(e):
        ents.append(e)
        by_id[e["id"]] = e
        return e

    for bid, b in enumerate(env.world.blocks):
        blk, f = b["block"], b["floor"]
        for e in block_entities(blk, f"b{bid}", b["x"], b["y"]):
            e["pos"][2] += f * FLOOR_H
            add(e)
        belts += block_belts(blk, f"b{bid}")

    own = end_owner(env)
    inward = INWARD[_side_of(env)]
    counter = [0]

    def piece(cls, cell, yaw):
        counter[0] += 1
        return add({"id": f"j{counter[0]}", "kind": "attachment", "class": cls,
                    "pos": cell_pos(*cell), "yaw": yaw})

    for item in sorted(env.entries):
        x, y = env.entries[item]
        add({"id": f"in_{item}", "kind": "container", "class": CONTAINER, "yaw": YAW_Y[inward],
                 "pos": [x + 0.5 - inward[0] * CONTAINER_GAP, y + 0.5 - inward[1] * CONTAINER_GAP, 0.0],
                 "fill": {"item": item, "amount": FILL}})
    ex, ey = env.exit
    out_dir = inward
    add({"id": "out", "kind": "container", "class": CONTAINER, "yaw": YAW_Y[out_dir],
         "pos": [ex + 0.5 + out_dir[0] * CONTAINER_GAP, ey + 0.5 + out_dir[1] * CONTAINER_GAP, 0.0]})

    ch = chains(env)
    path_of = {(L[1], L[2]): p for L, p in routes["paths"].items()}
    rate_of = {(L[1], L[2]): L[3] for L in routes["paths"]}

    for item in sorted(ch):
        cells, P = ch[item]["cells"], ch[item]["n_prod"]
        last = len(cells) - 1
        junction = {}
        for i, cell in enumerate(cells):
            if 1 <= i < P or P <= i < last:
                junction[i] = cell

        jinfo = {}
        for i, cell in junction.items():
            p_in = path_of[(cells[i - 1], cell)]
            p_out = path_of[(cell, cells[i + 1])]
            d_in, d_out = step(p_in[-2], p_in[-1]), step(p_out[0], p_out[1])
            o = own[cell]
            d_blk = block_side(env, cell, o[0])
            blk_ent = by_id[_block_port(env, o)[0]]
            if i < P:
                cls, yaw = MERGER, YAW_X[d_out]
                j = piece(cls, cell, yaw)
                trunk_in = port_with(cls, yaw, (-d_in[0], -d_in[1]), "in")
                blk_port = port_with(cls, yaw, d_blk, "in")
                bp_id, bp_port = _block_port(env, o)
                belts.append(belt(belt_class(env.world.blocks[o[0]]["block"]["output"]["rate"]),
                                  bp_id, bp_port, j["id"], blk_port))
                jinfo[i] = ((j["id"], trunk_in), (j["id"], "Output1"))
            else:
                cls, yaw = SPLITTER, YAW_X[d_in]
                j = piece(cls, cell, yaw)
                trunk_out = port_with(cls, yaw, d_out, "out")
                blk_port = port_with(cls, yaw, d_blk, "out")
                bp_id, bp_port = _block_port(env, o)
                belts.append(belt(belt_class(env.world.blocks[o[0]]["block"]["inputs"][o[1]]["rate"]),
                                  j["id"], blk_port, bp_id, bp_port))
                jinfo[i] = ((j["id"], "Input1"), (j["id"], trunk_out))

        for i in range(last):
            a, b = cells[i], cells[i + 1]
            path = path_of[(a, b)]
            cls_belt = belt_class(rate_of[(a, b)])

            if i in jinfo:
                src = jinfo[i][1]
                natural = step(path[0], path[1])
            else:
                o = own[a]
                if o[0] == "raw":
                    src = (f"in_{o[1]}", "Output1")
                    natural = inward
                else:
                    src = _block_port(env, o)
                    d = block_side(env, a, o[0])
                    natural = (-d[0], -d[1])
            if i + 1 in jinfo:
                dst = jinfo[i + 1][0]
                arrive = step(path[-2], path[-1])
            else:
                o = own[b]
                if o[0] == "exit":
                    dst = ("out", "Input0")
                    arrive = out_dir
                else:
                    dst = _block_port(env, o)
                    arrive = block_side(env, b, o[0])

            dirs = [step(p, q) for p, q in zip(path, path[1:])]
            corners = []
            if dirs[0] != natural:
                corners.append((path[0], natural, dirs[0]))
            run = 0
            for k in range(1, len(path) - 1):
                run += 1
                if dirs[k - 1] != dirs[k] or run >= MAX_RUN:
                    corners.append((path[k], dirs[k - 1], dirs[k]))
                    run = 0
            if dirs[-1] != arrive:
                corners.append((path[-1], dirs[-1], arrive))

            prev = src
            for cell, d_in, d_out in corners:
                yaw = YAW_X[d_out]
                j = piece(MERGER, cell, yaw)
                belts.append(belt(cls_belt, prev[0], prev[1], j["id"],
                                  port_with(MERGER, yaw, (-d_in[0], -d_in[1]), "in")))
                prev = (j["id"], "Output1")
            belts.append(belt(cls_belt, prev[0], prev[1], dst[0], dst[1]))

    for e in belts:
        a = port_pos(by_id[e["from"]], e["from_port"])
        b = port_pos(by_id[e["to"]], e["to_port"])
        if math.dist(a, b) < MIN_BELT - 1e-9:
            raise ValueError(f"belt {e['id']} would be {math.dist(a, b):.2f} m long (turns too close)")

    ents = floor_foundations(ents) + ents
    return {"schema": "satai.factoryplan", "version": 1, "name": name, "entities": ents + belts}

def _side_of(env):
    """Finds the site side of raw entries"""
    F, W, D = env.world.owner.shape
    exits = {0: (W - 1, D // 2), 1: (W // 2, D - 1), 2: (0, D // 2), 3: (W // 2, 0)}
    for side, cell in exits.items():
        if tuple(env.exit) == cell:
            return side
    raise ValueError(f"exit {env.exit} is not in the middle of a side")

def _block_port(env, owner):
    bid, k = owner
    blk = env.world.blocks[bid]["block"]
    if k == "out":
        return f"b{bid}_g{blk['n'] - 1}", "Output1"
    return f"b{bid}_s{k}_0", "Input1"


def main():
    ap = argparse.ArgumentParser(description="Export a saved layout to a FactoryPlan.")
    ap.add_argument("layout")
    ap.add_argument("--out")
    ap.add_argument("--png")
    args = ap.parse_args()
    name = Path(args.layout).stem
    task, plan = load(args.layout)
    res = evaluate(task, plan)
    draw(res["env"], args.png or f"{name}.png", title=f"{name}  score {res['score']:.3f}", routes=res["routes"])
    fp = layout_plan(res["env"], res["routes"], name)
    problems = validate_plan(fp)
    if problems:
        raise SystemExit("plan is invalid:\n  " + "\n  ".join(problems))
    out = Path(args.out or f"tests/plans/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=1), encoding="utf-8")
    kinds = {}
    for e in fp["entities"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"score {res['score']:.3f}; wrote {out}: {kinds}; picture {args.png or name + '.png'}")

if __name__ == "__main__":
    main()