import numpy as np
 
from solver.gamedata import recipe_by_id
from layout.connect import ends_for
from layout.tasks import GRID_MAX, FLOORS, MAX_MACHINES
from layout.world import EMPTY as NO_OWNER
from layout.blocks import make_block
from layout.env import N_MAX, raw_items
from layout.queue import depths
from solver.graph import BELT_CAPACITY

# Grid channels
SITE, BLOCK, PORT = 0, 1, 2
NEED = 3 # Where input k comes from
MAX_INPUTS = 4
GIVE, SAME, OPEN = 7, 8, 9 # where its output must go / other outputs of the same item / outputs still wanted later
N_CH = 10

# Vector indexes
MACHINES = ["Build_AssemblerMk1_C", "Build_ConstructorMk1_C", "Build_FoundryMk1_C",
    "Build_ManufacturerMk1_C", "Build_SmelterMk1_C"]
V_MACH = 0 # one of 0-4 machine type of the next block
V_SLOT = 5 # 5-8, 1 if input slot k exists
V_W, V_D, V_NLIM, V_REM, V_CLOCK, V_TARGET = 9, 10, 11, 12, 13, 14 # next block info, some things scaled down to be in approx. range 0-1
V_QLEN, V_MLEFT = 15, 16 # queue recipes left and machines left
V_FREE = 17 # 17-19 free fraction per floor
V_TIGHT, V_WIDTH, V_DEPTH = 20, 21, 22 # area still needed / free area, site size
N_VEC = 23
QUEUE_SCALE = 32
TIGHT_CAP = 2.0

# Graph features
MAX_NODES = 40
MAX_EDGES = 64
G_KIND = 0 # 0-2, 1 if recipe / raw entry / output
G_MACH = 3 # 3-7 1 for machine type (recipes only)
G_TOTAL, G_PLACED, G_HEAD, G_DEPTH = 8, 9, 10, 11
G_POS = 12 # 12-14 mean of the node's cells
N_NODE = 15
DEPTH_SCALE = 10
OUT = "OUTPUT"

def paint(obs, ch, cells):
    """Sets every cell in a given list to 1 in an observation channel"""
    for f, x, y in cells:
        obs[ch, f, x, y] = 1

def base_layers(env, obs):
    """Fill SITE, BLOCK and PORT from env.world"""
    F, W, D = env.world.owner.shape
    obs[SITE, :F, :W, :D] = 1
    obs[BLOCK, :F, :W, :D] = env.world.owner != NO_OWNER
    obs[PORT, :F, :W, :D] = env.world.reserved

def head_layers(env, obs):
    """Fill NEED_k, GIVE, SAME, OPEN for the next block"""
    if len(env.queue) == 0:
        return
    
    rec = recipe_by_id[env.queue[0]["recipe"]]
    for k, item in enumerate(rec["inputs"]):
        prods = ends_for(env, item)[0]
        paint(obs, NEED + k, prods)

    out = list(rec["outputs"])[0]
    prods, _, cons, _ = ends_for(env, out)
    paint(obs, GIVE, cons)
    paint(obs, SAME, prods)

    wanted = set()
    for q in env.queue:
        wanted.update(recipe_by_id[q["recipe"]]["inputs"])
    for item in sorted(wanted):
        paint(obs, OPEN, ends_for(env, item)[0])

def grid_obs(env):
    """Makes the whole observation grid"""
    obs = np.zeros((N_CH, FLOORS, GRID_MAX, GRID_MAX), dtype=np.uint8)
    base_layers(env, obs)
    head_layers(env, obs)
    return obs

def free_cells(env):
    """Number of free cells on each floor"""
    free = (env.world.owner == NO_OWNER) & ~env.world.reserved
    return free.sum(axis=(1, 2))

def head_vec(env, vec):
    """Fill the next-block features"""
    if len(env.queue) == 0:
        return
    
    head = env.queue[0]
    b = env.block(1, 0)
    m_index = MACHINES.index(head["machine"])
    vec[V_MACH + m_index] = 1.0

    for k in range(len(b["inputs"])):
        vec[V_SLOT + k] = 1.0

    vec[V_W] = b["w"] / GRID_MAX
    vec[V_D] = b["d"] / GRID_MAX
    vec[V_NLIM] = env.n_limit() / N_MAX
    vec[V_REM] = head["remaining"] / MAX_MACHINES
    vec[V_CLOCK] = head["clock"]
    if b["output"]["item"] == env.graph["target"]:
        vec[V_TARGET] = 1.0
 
def queue_vec(env, vec):
    """Fill the queue and site features"""
    F, W, D = env.world.owner.shape
    free = free_cells(env)

    for f in range(F):
        vec[V_FREE + f] = free[f] / (W * D)

    need = 0
    left = 0
    for q in env.queue:
        b = make_block(q["recipe"], 1, q["clock"])
        need += q["remaining"] * b["w"] * b["d"]
        left += q["remaining"]

    vec[V_TIGHT] = min(need / max(free.sum(), 1), TIGHT_CAP)
    vec[V_QLEN] = len(env.queue) / QUEUE_SCALE
    vec[V_MLEFT] = left / MAX_MACHINES
    vec[V_WIDTH] = W / GRID_MAX
    vec[V_DEPTH] = D / GRID_MAX
 
def vec_obs(env):
    """Creates the final observation vector"""
    vec = np.zeros(N_VEC, dtype=np.float32)
    head_vec(env, vec)
    queue_vec(env, vec)
    return vec

def node_list(env):
    """Gets a list of node keys in order (depth then name, output at end)"""
    d = depths(env.graph)
    recipes = sorted(env.graph["nodes"], key=lambda r: (-d[r], r))
    return recipes + raw_items(env.graph) + [OUT]
 
def placed(env):
    """Makes a dict of all the machines placed per recipe and their ends"""
    out = {}
    for b in env.world.blocks:
        r = b["block"]["recipe"]
        info = out.setdefault(r, {"machines": 0, "ends": []})
        info["machines"] += b["block"]["n"]
        info["ends"].append((b["floor"], *b["ends"][-1]))
    return out
 
def node_feats(env, keys):
    """Creates a feature row per node"""
    feats = np.zeros((MAX_NODES, N_NODE), dtype=np.float32)
    d = depths(env.graph)
    pl = placed(env)
    head = env.queue[0]["recipe"] if env.queue else None

    for i, key in enumerate(keys):
        row = feats[i]
        if key == OUT:
            row[G_KIND + 2] = 1.0
            row[G_PLACED] = 1.0
            cells = [(0, *env.exit)]

        elif key in env.entries:
            row[G_KIND + 1] = 1.0
            row[G_PLACED] = 1.0
            cells = [(0, *env.entries[key])]

        else:
            node = env.graph["nodes"][key]
            total = len(node["clocks"])
            info = pl.get(key, {"machines": 0, "ends": []})

            row[G_KIND] = 1.0
            row[G_MACH + MACHINES.index(node["machine"])] = 1.0
            row[G_TOTAL] = total / MAX_MACHINES
            row[G_PLACED] = info["machines"] / total

            if key == head:
                row[G_HEAD] = 1.0

            row[G_DEPTH] = d[key] / DEPTH_SCALE
            cells = info["ends"]

        if cells:
            m = np.mean(cells, axis=0)
            row[G_POS] = m[0] / (FLOORS - 1)
            row[G_POS + 1] = m[1] / GRID_MAX
            row[G_POS + 2] = m[2] / GRID_MAX

    return feats

def edge_arrays(env, index):
    """Builds an array of edges"""
    pairs = np.zeros((MAX_EDGES, 2), dtype=np.int32)
    rate = np.zeros(MAX_EDGES, dtype=np.float32)

    for j, e in enumerate(env.graph["edges"]):
        src = e["item"] if e["from"] == "RAW" else e["from"]
        pairs[j] = (index[src], index[e["to"]])
        rate[j] = e["rate"] / BELT_CAPACITY[-1]

    return pairs, rate
    
def graph_obs(env):
    """{"nodes", "node_mask", "edges", "edge_rate", "edge_mask"}: the whole graph observation."""
    keys = node_list(env)
    index = {k: i for i, k in enumerate(keys)}

    pairs, rate = edge_arrays(env, index)
    n_mask = np.zeros(MAX_NODES, dtype=bool)
    n_mask[:len(keys)] = True

    e_mask = np.zeros(MAX_EDGES, dtype=bool)
    e_mask[:len(env.graph["edges"])] = True

    return {"nodes": node_feats(env, keys), "node_mask": n_mask, "edges": pairs, "edge_rate": rate, "edge_mask": e_mask}