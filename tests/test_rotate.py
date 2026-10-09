import math

from executor.build import rotate
from layout.blocks import make_block, rot_cell, rot_point, rotate_block
from layout.export import block_belts, block_entities
from layout.world import World, end_cells
from tests.test_blocks import check_export, supported_recipes

SMELT = ("Recipe_IngotIron_C", 2, 1.0)
ROTOR = ("Recipe_Rotor_C", 3, 5 / 6)


def geometry(b):
    """Everything rotation changes, as one comparable tuple."""
    return (b["w"], b["d"], b["rot"], b["machines"],
            [s["cell"] for s in b["splitters"]], [g["cell"] for g in b["mergers"]],
            end_cells(b))


def test_rot_point_and_cell():
    assert rot_cell(0, 0, 2) == (1, 0)
    assert rot_cell(2, 1, 2) == (0, 2)
    assert rot_cell(-1, 0, 2) == (1, -1)
    assert rot_cell(3, 1, 2) == (0, 3)
    assert rot_point(0, 0, 3, 2) == (2, 0)
    assert rot_point(3, 2, 3, 2) == (0, 3)
    assert rot_point(0.5, 0.5, 3, 2) == (1.5, 0.5)
    for c in range(-1, 4):
        for r in range(-1, 3):
            px, py = rot_point(c + 0.5, r + 0.5, 3, 2)
            assert rot_cell(c, r, 2) == (px - 0.5, py - 0.5), (c, r)


def test_rotate_smelters_by_hand():
    b = rotate_block(make_block(*SMELT), 1)
    assert (b["w"], b["d"], b["rot"]) == (20, 10, 1)
    assert b["machines"] == [(10.0, 2.5), (10.0, 7.5)]
    assert [s["cell"] for s in b["splitters"]] == [(17, 2), (17, 7)]
    assert [g["cell"] for g in b["mergers"]] == [(2, 2), (2, 7)]
    assert b["inputs"][0]["end"] == (17, -1)
    assert b["output"]["end"] == (2, 10)

    b = rotate_block(make_block(*SMELT), 2)
    assert (b["w"], b["d"], b["rot"]) == (10, 20, 2)
    assert b["machines"] == [(7.5, 10.0), (2.5, 10.0)]
    assert b["inputs"][0]["end"] == (10, 17)
    assert b["output"]["end"] == (-1, 2)


def test_rotate_keeps_everything_else():
    b = make_block(*ROTOR)
    r = rotate_block(b, 1)
    for key in ("recipe", "machine", "n", "clock"):
        assert r[key] == b[key]
    for a, c in zip(b["inputs"], r["inputs"]):
        assert (a["item"], a["rate"], a["row"]) == (c["item"], c["rate"], c["row"])
    for a, c in zip(b["splitters"], r["splitters"]):
        assert (a["item"], a["port"], a["input"], a["machine"]) == (c["item"], c["port"], c["input"], c["machine"])


def test_rotate_does_not_change_input():
    b = make_block(*ROTOR)
    before = geometry(b)
    rotate_block(b, 1)
    rotate_block(b, 3)
    assert geometry(b) == before


def test_rotate_round_trips():
    for rid in supported_recipes()[::7]:
        b = make_block(rid, 3, 1.0)
        assert geometry(rotate_block(b, 0)) == geometry(b), rid
        assert geometry(rotate_block(b, 4)) == geometry(b), rid
        assert geometry(rotate_block(rotate_block(b, 1), 3)) == geometry(b), rid
        assert geometry(rotate_block(rotate_block(b, 2), 2)) == geometry(b), rid
        assert geometry(rotate_block(b, 5)) == geometry(rotate_block(b, 1)), rid
        assert geometry(rotate_block(b, -1)) == geometry(rotate_block(b, 3)), rid


def test_rotated_cells_stay_inside():
    for rid in supported_recipes()[::5]:
        for k in range(4):
            b = rotate_block(make_block(rid, 2, 1.0), k)
            w, d = b["w"], b["d"]
            cells = [s["cell"] for s in b["splitters"]] + [g["cell"] for g in b["mergers"]]
            assert all(0 <= c < w and 0 <= r < d for c, r in cells), (rid, k)
            assert all(0 < x < w and 0 < y < d for x, y in b["machines"]), (rid, k)
            for c, r in end_cells(b):
                outside_x = c in (-1, w) and 0 <= r < d
                outside_y = r in (-1, d) and 0 <= c < w
                assert outside_x != outside_y, (rid, k, (c, r))


def test_entries_and_exit_face_the_right_way():
    sides = {0: ("left", "right"), 1: ("top", "bottom"), 2: ("right", "left"), 3: ("bottom", "top")}
    def side(cell, w, d):
        c, r = cell
        return {-1: "left", w: "right"}.get(c) or {-1: "top", d: "bottom"}[r]
    for k in range(4):
        b = rotate_block(make_block(*ROTOR), k)
        want_in, want_out = sides[k]
        assert all(side(i["end"], b["w"], b["d"]) == want_in for i in b["inputs"]), k
        assert side(b["output"]["end"], b["w"], b["d"]) == want_out, k


def test_export_is_a_rigid_turn():
    for recipe, n, clock in (SMELT, ROTOR):
        b = make_block(recipe, n, clock)
        flat = {e["id"]: e for e in block_entities(b, "p", 0.0, 0.0)}
        for k in range(4):
            r = rotate_block(b, k)
            shift = {0: (0, 0), 1: (b["d"], 0), 2: (b["w"], b["d"]), 3: (0, b["w"])}[k]
            turned = {e["id"]: e for e in block_entities(r, "p", 0.0, 0.0)}
            assert set(turned) == set(flat)
            for eid, e in flat.items():
                x, y = rotate(e["pos"][0], e["pos"][1], 90 * k)
                want = (x + shift[0], y + shift[1], e["pos"][2])
                got = turned[eid]["pos"]
                assert all(math.isclose(a, c, abs_tol=1e-9) for a, c in zip(got, want)), (recipe, k, eid)
                assert turned[eid]["yaw"] == 90 * k, (recipe, k, eid)
            assert block_belts(r, "p") == block_belts(b, "p")


def test_rotated_export_valid():
    for rid in supported_recipes()[::11]:
        for k in range(4):
            check_export(rotate_block(make_block(rid, 2, 1.0), k))


def test_world_with_rotated_block():
    b = rotate_block(make_block(*SMELT), 1)
    world = World(1, 20, 12)
    mask = world.fit_mask(b["w"], b["d"], end_cells(b))
    assert mask.shape == (1, 1, 3)
    assert list(mask[0, 0]) == [False, True, False]
    world.place(b, 0, 0, 1)
    assert world.blocks[0]["ends"] == [(17, 0), (2, 11)]
    assert world.reserved[0, 17, 0] and world.reserved[0, 2, 11]


if __name__ == "__main__":
    tests = [test_rot_point_and_cell, test_rotate_smelters_by_hand, test_rotate_keeps_everything_else,
             test_rotate_does_not_change_input, test_rotate_round_trips, test_rotated_cells_stay_inside,
             test_entries_and_exit_face_the_right_way, test_export_is_a_rigid_turn,
             test_rotated_export_valid, test_world_with_rotated_block]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")