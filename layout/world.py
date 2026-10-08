import numpy as np

EMPTY = -1

def end_cells(block):
    """The block's entry and exit cells in block coordinates"""
    cells = [inp["end"] for inp in block.get("inputs", [])]
    if "output" in block:
        cells.append(block["output"]["end"])
    return cells


class World:
    def __init__(self, floors, width, depth):
        """Creates a world class with an array of owners of each block"""

        self.owner = np.full((floors, width, depth), EMPTY, dtype=np.int32)
        self.reserved = np.zeros((floors, width, depth), dtype=bool)
        self.blocks = []

    def fit_mask(self, w, d, ends=()):
        """Checks all possible locations that a w x d block could fit"""

        F, X, Y = self.owner.shape
        blocked = (self.owner != EMPTY) | self.reserved
        occ = blocked.astype(np.int32) 
        S = np.zeros((F, X + 1, Y + 1), dtype=np.int32)
        S[:, 1:, 1:] = occ.cumsum(axis=1).cumsum(axis=2)

        used = S[:, w:, d:] - S[:, :-w, d:] - S[:, w:, :-d] + S[:, :-w, :-d]
        mask = used == 0
        if mask.size == 0:
            return mask 

        free = np.zeros((F, X + 2, Y + 2), dtype=bool)
        free[:, 1:-1, 1:-1] = ~blocked
        nx, ny = mask.shape[1], mask.shape[2]
        for ex, ey in ends:
            mask &= free[:, ex + 1:ex + 1 + nx, ey + 1:ey + 1 + ny]
        return mask

    def place(self, block, f, x, y):
        """Mark the block's w x d cells with a new id. Raise ValueError if it is out of bounds or overlaps."""

        F, X, Y = self.owner.shape
        w, d = block["w"], block["d"]
        ends = end_cells(block)

        if not (0 <= f < F and 0 <= x and x + w <= X and 0 <= y and y + d <= Y):
            raise ValueError(f"block {w}x{d} at floor {f} ({x}, {y}) is out of bounds")
        
        if not self.fit_mask(w, d, ends)[f, x, y]:
            raise ValueError(f"block {w}x{d} at floor {f} ({x}, {y}) overlaps or blocks an end")
        
        bid = len(self.blocks)
        self.owner[f, x:x + w, y:y + d] = bid
        cells = [(x + ex, y + ey) for ex, ey in ends]
        for cx, cy in cells:
            self.reserved[f, cx, cy] = True 
        self.blocks.append({"block": block, "floor": f, "x": x, "y": y, "ends": cells})

        return bid