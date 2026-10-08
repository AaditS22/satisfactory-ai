import random

import numpy as np

from layout.blocks import make_block
from layout.world import EMPTY, World, end_cells

def fake_block(w, d, ins=(), out=None):
    b = {"w": w, "d": d, "inputs": [{"end": e} for e in ins]}
    if out is not None:
        b["output"] = {"end": out}
    return b

def expect_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args[1:]} should raise ValueError")

def test_new_world():
    world = World(3, 6, 4)
    assert world.owner.shape == (3, 6, 4)
    assert (world.owner == EMPTY).all()
    assert world.blocks == []
    assert world.reserved.shape == (3, 6, 4)
    assert not world.reserved.any()

def test_fit_mask_empty():
    mask = World(2, 6, 4).fit_mask(2, 2)
    assert mask.shape == (2, 5, 3)
    assert mask.dtype == bool
    assert mask.all()

def test_fit_mask_worked_example():
    world = World(1, 6, 4)
    world.owner[0, 2:4, 0:2] = 0
    mask = world.fit_mask(2, 2)
    assert mask.sum() == 9
    expected = np.array([[1, 1, 1],     # x = 0
                         [0, 0, 1],     # x = 1: overlaps at y = 0, 1
                         [0, 0, 1],     # x = 2
                         [0, 0, 1],     # x = 3
                         [1, 1, 1]],    # x = 4
                        dtype=bool)
    assert (mask[0] == expected).all()

def test_fit_mask_edges():
    world = World(1, 6, 4)
    assert world.fit_mask(6, 4).shape == (1, 1, 1)       # exactly the floor: one position
    assert world.fit_mask(6, 4).all()
    assert world.fit_mask(1, 1).shape == (1, 6, 4)       # one per cell
    for w, d in [(7, 4), (6, 5), (20, 20)]:              # too big: no positions at all
        assert not world.fit_mask(w, d).any(), (w, d)
    world.owner[0, 5, 3] = 0                             # one corner cell taken
    assert not world.fit_mask(6, 4).any()

def test_fit_mask_floors_independent():
    world = World(2, 6, 4)
    world.owner[0] = 0                                   # ground floor full
    mask = world.fit_mask(3, 3)
    assert not mask[0].any()
    assert mask[1].all()

def test_fit_mask_real_block():
    b = make_block("Recipe_Rotor_C", 3, 5 / 6)           # 27 x 30
    mask = World(2, 40, 40).fit_mask(b["w"], b["d"])
    assert mask.shape == (2, 14, 11)

def test_place():
    world = World(2, 10, 8)
    assert world.place(fake_block(3, 2), 0, 1, 4) == 0
    assert world.place(fake_block(2, 2), 1, 1, 4) == 1   # same spot, other floor
    assert world.place(fake_block(4, 4), 0, 6, 4) == 2   # touching block 0's right side
    assert (world.owner[0, 1:4, 4:6] == 0).all()
    assert (world.owner[1, 1:3, 4:6] == 1).all()
    assert (world.owner[0, 6:10, 4:8] == 2).all()
    assert (world.owner != EMPTY).sum() == 6 + 4 + 16    # nothing else marked
    assert world.blocks[0] == {"block": fake_block(3, 2), "floor": 0, "x": 1, "y": 4, "ends": []}
    assert [p["floor"] for p in world.blocks] == [0, 1, 0]

def test_place_refuses():
    world = World(2, 10, 8)
    world.place(fake_block(3, 3), 0, 2, 2)
    before = world.owner.copy()
    reserved = world.reserved.copy()
    expect_error(world.place, fake_block(2, 2), 0, 4, 4)     # overlaps one cell (4, 4)
    expect_error(world.place, fake_block(2, 2), 0, 9, 0)     # sticks out on the right
    expect_error(world.place, fake_block(2, 2), 0, 0, 7)     # sticks out at the back
    expect_error(world.place, fake_block(2, 2), 0, -1, 0)    # negative x
    expect_error(world.place, fake_block(2, 2), 0, 0, -1)    # negative y
    expect_error(world.place, fake_block(2, 2), 2, 0, 0)     # no floor 2
    expect_error(world.place, fake_block(2, 2), -1, 0, 0)    # negative floor
    assert (world.owner == before).all()                     # refusals change nothing
    assert (world.reserved == reserved).all()
    assert len(world.blocks) == 1

def test_end_cells():
    b = make_block("Recipe_Rotor_C", 3, 5 / 6)               # 27 x 30, rows 6 and 2, output row 27
    assert end_cells(b) == [(-1, 6), (-1, 2), (27, 27)]
    assert end_cells(fake_block(2, 2)) == []


def test_end_mask_worked_example():
    mask = World(1, 6, 4).fit_mask(2, 2, [(-1, 0)])
    assert mask.shape == (1, 5, 3)
    assert not mask[0, 0].any()
    assert mask[0, 1:].all()
    mask = World(1, 6, 4).fit_mask(2, 2, [(-1, 0), (2, 1)])
    assert not mask[0, 0].any() and not mask[0, 4].any()
    assert mask[0, 1:4].all()
    assert mask.sum() == 9


def test_end_inside_other_block():
    world = World(1, 10, 4)
    world.place(fake_block(2, 4), 0, 0, 0)
    mask = world.fit_mask(2, 2, [(-1, 0)])
    assert not mask[0, 2].any()                              # entry would be at x = 1: taken
    assert mask[0, 3].all()                                  # one free cell gap is enough
    expect_error(world.place, fake_block(2, 2, ins=[(-1, 0)]), 0, 2, 0)
    world.place(fake_block(2, 2, ins=[(-1, 0)]), 0, 3, 0)


def test_cover_other_blocks_end():
    world = World(1, 10, 4)
    world.place(fake_block(2, 2, ins=[(-1, 1)]), 0, 4, 0)
    assert world.reserved[0, 3, 1] and world.reserved.sum() == 1
    assert world.blocks[0]["ends"] == [(3, 1)]
    assert world.owner[0, 3, 1] == EMPTY                     # reserved, but not owned
    mask = world.fit_mask(2, 2)
    assert not mask[0, 2, 0] and not mask[0, 2, 1] and not mask[0, 3, 0] and not mask[0, 3, 1]
    assert mask[0, 2, 2] and mask[0, 1, 0]
    expect_error(world.place, fake_block(2, 2), 0, 2, 0)
    world.place(fake_block(2, 2), 0, 1, 0)                   # ends at x = 2: fine


def test_end_on_other_blocks_end():
    world = World(1, 10, 4)
    world.place(fake_block(2, 2, out=(2, 0)), 0, 0, 0)       # exit reserved at (2, 0)
    assert not world.fit_mask(2, 2, [(-1, 0)])[0, 3, 0]      # entry would be at (2, 0)
    assert world.fit_mask(2, 2, [(-1, 0)])[0, 4, 0]

def test_end_mask_real_block():
    b = make_block("Recipe_Rotor_C", 3, 5 / 6)               # needs 1 cell left and right
    assert not World(1, 27, 40).fit_mask(b["w"], b["d"], end_cells(b)).any()
    assert not World(1, 28, 40).fit_mask(b["w"], b["d"], end_cells(b)).any()
    mask = World(1, 29, 40).fit_mask(b["w"], b["d"], end_cells(b))
    assert mask.shape == (1, 3, 11)
    assert mask[0, 1].all() and not mask[0, 0].any() and not mask[0, 2].any()
    world = World(1, 29, 40)
    world.place(b, 0, 1, 5)
    assert world.blocks[0]["ends"] == [(0, 11), (0, 7), (28, 32)]
    assert world.reserved.sum() == 3

def random_block(rng):
    w, d = rng.randint(1, 4), rng.randint(1, 4)
    ins = [(-1, y) for y in rng.sample(range(d), rng.randint(0, d))]
    out = (w, rng.randrange(d)) if rng.random() < 0.7 else None
    return fake_block(w, d, ins, out)


def fits(world, b, f, x, y):
    F, X, Y = world.owner.shape
    taken = (world.owner != EMPTY) | world.reserved
    if taken[f, x:x + b["w"], y:y + b["d"]].any():
        return False
    for ex, ey in end_cells(b):
        cx, cy = x + ex, y + ey
        if not (0 <= cx < X and 0 <= cy < Y) or taken[f, cx, cy]:
            return False
    return True


def test_mask_matches_place():
    rng = random.Random(0)
    for _ in range(200):
        world = World(2, rng.randint(5, 12), rng.randint(5, 12))
        F, X, Y = world.owner.shape
        for _ in range(rng.randint(0, 8)):
            b = random_block(rng)
            f = rng.randrange(F)
            free = np.argwhere(world.fit_mask(b["w"], b["d"], end_cells(b))[f])
            if len(free):
                x, y = free[rng.randrange(len(free))]
                world.place(b, f, int(x), int(y))
        b = random_block(rng)
        mask = world.fit_mask(b["w"], b["d"], end_cells(b))
        for f in range(F):
            for x in range(X - b["w"] + 1):
                for y in range(Y - b["d"] + 1):
                    assert mask[f, x, y] == fits(world, b, f, x, y), (f, x, y, b)


def test_old_mask_matches_place():
    rng = random.Random(0)
    for _ in range(30):
        world = World(2, rng.randint(5, 12), rng.randint(5, 12))
        F, X, Y = world.owner.shape
        for _ in range(rng.randint(0, 6)):
            f, w, d = rng.randrange(F), rng.randint(1, 4), rng.randint(1, 4)
            free = np.argwhere(world.fit_mask(w, d)[f])
            if len(free):
                x, y = free[rng.randrange(len(free))]
                world.place(fake_block(w, d), f, int(x), int(y))
        w, d = rng.randint(1, 5), rng.randint(1, 5)
        mask = world.fit_mask(w, d)
        for f in range(F):
            for x in range(X - w + 1):
                for y in range(Y - d + 1):
                    free = (world.owner[f, x:x + w, y:y + d] == EMPTY).all()
                    assert mask[f, x, y] == free, (f, x, y, w, d)


if __name__ == "__main__":
    tests = [test_new_world, test_fit_mask_empty, test_fit_mask_worked_example, test_fit_mask_edges,
             test_fit_mask_floors_independent, test_fit_mask_real_block, test_place, test_place_refuses,
             test_end_cells, test_end_mask_worked_example, test_end_inside_other_block,
             test_cover_other_blocks_end, test_end_on_other_blocks_end, test_end_mask_real_block,
             test_mask_matches_place, test_old_mask_matches_place]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} passed")