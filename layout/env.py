import numpy as np

from solver.graph import build_graph
from layout.queue import make_queue, max_block, take
from layout.blocks import make_block, rotate_block
from layout.world import World, end_cells

N_MAX = 16
MAX_TIER = 6

def raw_items(graph):
    """Sorted list of the raw items the factory needs"""
    items = set()

    for edge in graph["edges"]:
        if edge["from"] == "RAW":
            items.add(edge["item"])

    return sorted(items)

def edge_cells(side, k, width, depth):
    """k cells spread evenly along one side of the ground floor"""
    x = (False, None)
    y = (False, None)
    v = 0

    match side:
        case 0:
            x = (True, 0)
            v = depth
        case 1:
            y = (True, 0)
            v = width
        case 2:
            x = (True, width - 1)
            v = depth
        case 3:
            y = (True, depth - 1)
            v = width

    points = []
    for i in range(k):
        var_val = (i + 1) * v // (k + 1)
        nx = x[1] if x[0] else var_val
        ny = y[1] if y[0] else var_val
        points.append((nx, ny))

    return points

class LayoutEnv:
    def reset(self, task):
        """Start a new episode"""
        self.graph = build_graph(task["target"], task["rate"], alternates=task.get("alternates", set()))
        self.queue = make_queue(self.graph)
        self.world = World(task["floors"], task["width"], task["depth"])

        items = raw_items(self.graph)
        cells = edge_cells(task["side"], len(items), task["width"], task["depth"])
        entries = {}
        for i, item in enumerate(items):
            entries[item] = cells[i]
            self.world.reserved[0, cells[i][0], cells[i][1]] = True
        self.entries = entries
        
        opposite = (task["side"] + 2) % 4
        self.exit = edge_cells(opposite, 1, task["width"], task["depth"])[0]
        x, y = self.exit
        self.world.reserved[0, x, y] = True

        self.done = False
        self.failed = False

        if not self.n_mask().any():
            self.done = True
            self.failed = True

    def block(self, n, rot):
        """The block of n machines of the queue head turned rot turns"""
        head = self.queue[0]

        blk = make_block(head["recipe"], n, head["clock"])
        return rotate_block(blk, rot)

    def n_limit(self):
        """Largest N the rules allow for the queue head"""
        head = self.queue[0]
        return min(head["remaining"], max_block(head, MAX_TIER), N_MAX)

    def pos_mask(self, n):
        """Checks where a block of n machines could go (for each rotation)"""
        F, X, Y = self.world.owner.shape
        out = np.zeros((4, F, X, Y), dtype=bool)

        for rot in range(4):
            b = self.block(n, rot)
            m = self.world.fit_mask(b["w"], b["d"], end_cells(b))
            out[rot, :, :m.shape[1], :m.shape[2]] = m

        return out
    
    def n_mask(self):
        """Checks each block of size n, true if it fits somewhere"""
        mask = np.zeros(N_MAX + 1, dtype=bool)

        for n in range(1, self.n_limit() + 1):
            mask[n] = self.pos_mask(n).any()

        return mask
    
    def step(self, n, rot, f, x, y):
        """Place one block. Returns 1.0 on finishing, -1.0 if nothing fits any more, else 0.0"""
        if self.done:
            raise ValueError("Already finished")

        F, X, Y = self.world.owner.shape
        if not (1 <= n <= self.n_limit() and 0 <= rot < 4 and
                0 <= f < F and 0 <= x < X and 0 <= y < Y):
            raise ValueError("One of the inputs is invalid")

        if not self.pos_mask(n)[rot, f, x, y]:
            raise ValueError("Position is invalid")

        b = self.block(n, rot)
        take(self.queue, n, MAX_TIER)
        self.world.place(b, f, x, y)

        if len(self.queue) == 0:
            self.done = True
            return 1.0

        if not self.n_mask().any():
            self.done = True
            self.failed = True
            return -1.0

        return 0.0