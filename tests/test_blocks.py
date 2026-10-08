import math

from executor.plan import validate_plan
from layout.blocks import CATALOG, SNAP, footprint, make_block
from layout.export import MERGER, SPLITTER, belt_class, block_belts, block_entities
from solver.gamedata import recipe_by_id

SMELTER = "Build_SmelterMk1_C"
CONSTRUCTOR = "Build_ConstructorMk1_C"
ASSEMBLER = "Build_AssemblerMk1_C"
FOUNDRY = "Build_FoundryMk1_C"
MANUFACTURER = "Build_ManufacturerMk1_C"
MACHINES = [SMELTER, CONSTRUCTOR, ASSEMBLER, FOUNDRY, MANUFACTURER]

SIZES = [1, 2, 5]
CLOCKS = [1.0, 0.5, 5 / 6, 2.5]

def supported_recipes():
    """Recipes a block can be made for: a machine with snap rules and one output."""
    return sorted(rid for rid, r in recipe_by_id.items()
                  if r["machine"] in MACHINES and len(r["outputs"]) == 1)

def ports(cls, direction):
    return {p["name"]: p for p in CATALOG[cls]["ports"] if p["dir"] == direction}

def rows(cells):
    return sorted({r for _, r in cells})

def test_footprint():
    assert footprint(SMELTER) == (5, 10, 2.5, 5.0)
    assert footprint(CONSTRUCTOR) == (9, 10, 4.5, 5.0)
    assert footprint(ASSEMBLER) == (9, 16, 4.5, 8.5)
    assert footprint(FOUNDRY) == (11, 10, 5.5, 5.5)
    assert footprint(MANUFACTURER) == (19, 20, 9.5, 11.0)


def test_smelter_block():
    b = make_block("Recipe_IngotIron_C", 2, 1.0)
    assert (b["machine"], b["n"], b["clock"]) == (SMELTER, 2, 1.0)
    assert (b["w"], b["d"]) == (10, 20)
    assert b["machines"] == [(2.5, 10.0), (7.5, 10.0)]
    assert [s["cell"] for s in b["splitters"]] == [(2, 2), (7, 2)]
    assert [m["cell"] for m in b["mergers"]] == [(2, 17), (7, 17)]
    assert len(b["inputs"]) == 1
    assert b["inputs"][0]["item"] == "Desc_OreIron_C"
    assert b["inputs"][0]["row"] == 2
    assert math.isclose(b["inputs"][0]["rate"], 60)
    assert b["output"]["item"] == "Desc_IronIngot_C"
    assert b["output"]["row"] == 17
    assert math.isclose(b["output"]["rate"], 60)
    assert b["inputs"][0]["end"] == (-1, 2)
    assert b["output"]["end"] == (10, 17)

def test_rotor_block():
    b = make_block("Recipe_Rotor_C", 3, 5 / 6)
    assert (b["w"], b["d"]) == (27, 30)
    assert b["machines"] == [(4.5, 17.5), (13.5, 17.5), (22.5, 17.5)]
    rod, screw = b["inputs"]
    assert (rod["item"], rod["row"]) == ("Desc_IronRod_C", 6)
    assert (screw["item"], screw["row"]) == ("Desc_IronScrew_C", 2)
    assert math.isclose(rod["rate"], 50)     # 20/min x 3 x 5/6
    assert math.isclose(screw["rate"], 250)  # 100/min x 3 x 5/6
    assert b["output"]["row"] == 27
    assert math.isclose(b["output"]["rate"], 10)  # 4/min x 3 x 5/6
    rod_cells = [s["cell"] for s in b["splitters"] if s["input"] == 0]
    screw_cells = [s["cell"] for s in b["splitters"] if s["input"] == 1]
    assert rod_cells == [(6, 6), (15, 6), (24, 6)]
    assert screw_cells == [(2, 2), (11, 2), (20, 2)]
    assert [m["cell"] for m in b["mergers"]] == [(4, 27), (13, 27), (22, 27)]


def test_manufacturer_block():
    rid = next(r for r in supported_recipes()
               if recipe_by_id[r]["machine"] == MANUFACTURER and len(recipe_by_id[r]["inputs"]) == 4)
    b = make_block(rid, 1, 1.0)
    assert (b["w"], b["d"]) == (19, 42)
    assert [i["row"] for i in b["inputs"]] == [14, 10, 6, 2]
    assert b["machines"] == [(9.5, 28.0)]
    assert b["output"]["row"] == 39


def test_belt_count_and_tiers():
    belts = block_belts(make_block("Recipe_IngotIron_C", 2, 1.0), "t")
    assert len(belts) == 6
    belts = block_belts(make_block("Recipe_Rotor_C", 3, 5 / 6), "t")
    assert len(belts) == 15
    by_id = {e["id"]: e["class"] for e in belts}
    assert by_id["belt_t_s0_0_t_s0_1"] == "Build_ConveyorBeltMk1_C"  # rod 50/min
    assert by_id["belt_t_s1_0_t_s1_1"] == "Build_ConveyorBeltMk3_C"  # screw 250/min
    assert by_id["belt_t_g0_t_g1"] == "Build_ConveyorBeltMk1_C"      # rotor 10/min

def test_entities_offset():
    b = make_block("Recipe_IngotIron_C", 2, 1.0)
    ents = {e["id"]: e for e in block_entities(b, "t", 100.0, 200.0)}
    assert set(ents) == {"t_m0", "t_m1", "t_s0_0", "t_s0_1", "t_g0", "t_g1"}
    assert ents["t_m1"]["pos"] == [107.5, 210.0, 0.0]
    assert ents["t_m1"]["recipe"] == "Recipe_IngotIron_C"
    assert ents["t_m1"]["clock"] == 1.0
    assert ents["t_s0_0"]["pos"] == [102.5, 202.5, 1.0]
    assert ents["t_g1"]["pos"] == [107.5, 217.5, 1.0]

def check_geometry(b):
    """Rules every block must obey. Returns nothing; asserts with a message."""
    tag = f"{b['recipe']} n={b['n']}"
    recipe = recipe_by_id[b["recipe"]]
    mw, md, cx, cy = footprint(b["machine"])
    n, k = b["n"], len(recipe["inputs"])
    top = 4 * k + 1   
    bottom = top + md - 1   
    in_ports, out_ports = ports(b["machine"], "in"), ports(b["machine"], "out")

    assert (b["w"], b["d"]) == (n * mw, 4 * k + md + 6), tag
    assert b["machines"] == [(m * mw + cx, top + cy) for m in range(n)], tag

    in_rows = [i["row"] for i in b["inputs"]]
    assert in_rows == [4 * (k - 1 - j) + 2 for j in range(k)], tag
    assert min(in_rows) == 2, tag 
    assert top - max(in_rows) == 3, tag   

    out = b["output"]["row"]
    assert out - bottom == 3 and b["d"] - 1 - out == 2, tag

    assert sorted((s["input"], s["machine"]) for s in b["splitters"]) == \
        [(j, m) for j in range(k) for m in range(n)], tag
    assert sorted(g["machine"] for g in b["mergers"]) == list(range(n)), tag

    cells = [s["cell"] for s in b["splitters"]] + [g["cell"] for g in b["mergers"]]
    assert len(set(cells)) == len(cells), f"{tag}: two attachments on one cell"
    for c, r in cells:
        assert 0 <= c < b["w"] and 0 <= r < b["d"], f"{tag}: {(c, r)} out of bounds"

    for s in b["splitters"]:
        x, _ = b["machines"][s["machine"]]
        port = in_ports[s["port"]]
        c, r = s["cell"]
        assert r == in_rows[s["input"]], tag
        assert s["item"] == b["inputs"][s["input"]]["item"], tag
        assert s["machine"] * mw <= c < (s["machine"] + 1) * mw, tag
        assert c + 0.5 == x + port["pos"][0], f"{tag}: splitter not in line with {s['port']}"

    for m in range(n):
        used = [s["port"] for s in b["splitters"] if s["machine"] == m]
        assert len(set(used)) == k, f"{tag}: machine {m} reuses a port"

    for j in range(k):
        xs = [s["cell"][0] for s in sorted(b["splitters"], key=lambda s: s["machine"]) if s["input"] == j]
        assert xs == sorted(xs) and len(set(xs)) == n, tag

    (out_name, out_port), = out_ports.items()
    for g in sorted(b["mergers"], key=lambda g: g["machine"]):
        x, _ = b["machines"][g["machine"]]
        c, r = g["cell"]
        assert r == out and g["port"] == out_name, tag
        assert c + 0.5 == x + out_port["pos"][0], f"{tag}: merger not in line with output"

    for i in b["inputs"]:
        assert i["end"] == (-1, i["row"]), tag
        assert "ends" not in i, tag
    assert b["output"]["end"] == (b["w"], out), tag
    assert "ends" not in b["output"], tag
    first = [s["cell"][0] for s in b["splitters"] if s["machine"] == 0]
    last = [g["cell"][0] for g in b["mergers"] if g["machine"] == n - 1]
    assert all(i["end"][0] < c for i, c in zip(b["inputs"], first)), tag
    assert b["output"]["end"][0] > last[0], tag

    assert [i["item"] for i in b["inputs"]] == list(recipe["inputs"]), tag
    for i in b["inputs"]:
        assert math.isclose(i["rate"], recipe["inputs"][i["item"]] * n * b["clock"]), tag
    assert b["output"]["item"] == next(iter(recipe["outputs"])), tag
    assert math.isclose(b["output"]["rate"], recipe["outputs"][b["output"]["item"]] * n * b["clock"]), tag


def check_export(b):
    """Entities and belts form a valid, fully wired plan."""
    tag = f"{b['recipe']} n={b['n']}"
    n, k = b["n"], len(b["inputs"])
    ents = block_entities(b, "p", 16.0, 32.0)
    belts = block_belts(b, "p")
    assert len(ents) == n + n * k + n, tag
    assert len(belts) == (k + 1) * (2 * n - 1), tag

    plan = {"schema": "satai.factoryplan", "version": 1, "name": "t", "entities": ents + belts}
    assert validate_plan(plan) == [], (tag, validate_plan(plan))

    cls = {e["id"]: e["class"] for e in ents}
    used = set()
    for e in belts:
        assert e["from"] in cls and e["to"] in cls, f"{tag}: belt to unknown id"
        assert e["from_port"] in ports(cls[e["from"]], "out"), f"{tag}: {e['id']} bad from_port"
        assert e["to_port"] in ports(cls[e["to"]], "in"), f"{tag}: {e['id']} bad to_port"
        for end in ((e["from"], e["from_port"]), (e["to"], e["to_port"])):
            assert end not in used, f"{tag}: port {end} used twice"
            used.add(end)

    for m in range(n):
        for p in CATALOG[b["machine"]]["ports"]:
            if p["dir"] == "out" or p["name"] in {s["port"] for s in b["splitters"]}:
                assert (f"p_m{m}", p["name"]) in used, f"{tag}: m{m}.{p['name']} unwired"
    open_in = [i for i, c in cls.items() if c == SPLITTER and (i, "Input1") not in used]
    open_out = [i for i, c in cls.items() if c == MERGER and (i, "Output1") not in used]
    assert sorted(open_in) == [f"p_s{j}_0" for j in range(k)], tag
    assert open_out == [f"p_g{n - 1}"], tag

    for e in belts:
        if cls[e["from"]] == MERGER or cls[e["to"]] == MERGER:
            assert e["class"] == belt_class(b["output"]["rate"]), tag
        else:
            j = int(e["from"].split("_")[1][1:])  # p_s{j}_{m}
            assert e["class"] == belt_class(b["inputs"][j]["rate"]), tag


def test_all_recipes():
    recipes = supported_recipes()
    assert len(recipes) > 100, len(recipes)
    for rid in recipes:
        for n in SIZES:
            for clock in CLOCKS:
                b = make_block(rid, n, clock)
                check_geometry(b)
                check_export(b)


if __name__ == "__main__":
    tests = [test_footprint, test_smelter_block, test_rotor_block, test_manufacturer_block,
             test_belt_count_and_tiers, test_entities_offset, test_all_recipes]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")