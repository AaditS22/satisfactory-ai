import json
from pathlib import Path

import numpy as np

from siting.terrain import STEP

MINER_BASE = {"impure": 30.0, "normal": 60.0, "pure": 120.0}
MINER_MULT = 2.0    

def load_nodes(save_name="SatAI_Test", root=".terrain", bridge=None):
    path = Path(root) / save_name / "nodes.json"
    if not path.exists():
        if bridge is None:
            from executor.bridge import Bridge
            bridge = Bridge()
        nodes = [n for n in bridge.resource_nodes() if n["type"] == "node" and n["form"] == "solid"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nodes, indent=1))
    return json.loads(path.read_text())

def node_rate(node):
    return MINER_BASE[node["purity"]] * MINER_MULT

def _supply_cost(dist, rates, demand):
    order = np.argsort(dist, axis=-1)   
    d = np.take_along_axis(dist, order, axis=-1)
    cap = rates[order]
    before = np.cumsum(cap, axis=-1) - cap     
    take = np.clip(demand - before, 0.0, cap)   
    return (take * d).sum(axis=-1), order, take

def transport_cost(xs, ys, nodes, demand):
    X = np.asarray(xs, dtype=float)[:, None] 
    Y = np.asarray(ys, dtype=float)[None, :] 
    total = np.zeros((X.shape[0], Y.shape[1]))
    for resource, amount in demand.items():
        mine = [n for n in nodes if n["resource"] == resource and not n["occupied"]]
        rates = np.array([node_rate(n) for n in mine])
        if rates.sum() < amount:
            return np.full(total.shape, np.inf)
        pos = np.array([n["pos"][:2] for n in mine])
        dist = np.hypot(X[..., None] - pos[:, 0], Y[..., None] - pos[:, 1])  
        total += _supply_cost(dist, rates, amount)[0]
    return total

def transport_map(terrain, nodes, demand):
    nx, ny = terrain.z.shape
    return transport_cost(terrain.x0 + np.arange(nx) * STEP, terrain.y0 + np.arange(ny) * STEP, nodes, demand)

def pick_nodes(centre, nodes, demand):
    used = []
    for resource, amount in demand.items():
        mine = [n for n in nodes if n["resource"] == resource and not n["occupied"]]
        rates = np.array([node_rate(n) for n in mine])
        dist = np.array([np.hypot(n["pos"][0] - centre[0], n["pos"][1] - centre[1]) for n in mine])
        _, order, take = _supply_cost(dist, rates, amount)
        used += [(mine[i], float(t), float(dist[i])) for i, t in zip(order, take) if t > 0]
    return used