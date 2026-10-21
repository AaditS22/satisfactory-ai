import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from layout import obs as O
from layout.env import N_MAX, LayoutEnv
from layout.gym_env import POS_SHAPE, N_POS, observe
from layout.tasks import random_task

IN_CH = O.N_CH * POS_SHAPE[1]  # Number of input maps initially
C1, C2, C3 = 32, 64, 128  # Number of desired maps for 48x48, 24x24, and 12x12 sized inputs
GRAPH_DIM = 64
HID = 128
SUMMARY_IN = C3 + GRAPH_DIM + O.N_VEC + 2  # Number of total inputs for map summary: 128 map  + 64 graph + 23 vec + 2 stage
N_MAPS = POS_SHAPE[0] * POS_SHAPE[1]  # Number of maps is the 4 rotations * 3 floors, 12 maps

def flatten_floors(grid):
    """uint8 (B, 10, 3, 96, 96) -> float32 (B, 30, 96, 96). Map c * 3 + f = channel c on floor f."""
    B = grid.shape[0]
    return grid.float().reshape(B, IN_CH, grid.shape[3], grid.shape[4])

class ConvBlock(nn.Module):
    """Performs two 3x3 convolutions, each followed by ReLU, without changing map size"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class GridEncoder(nn.Module):
    """Shrinks the 30 x 96 x 96 grid in 4 halving steps and keeps the in-between maps for the decoder"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(IN_CH, C1, kernel_size=3, padding=1, stride=2)
        self.block1 = ConvBlock(C1, C1)
        self.down1 = nn.Conv2d(C1, C2, kernel_size=3, padding=1, stride=2)
        self.block2 = ConvBlock(C2, C2)
        self.down2 = nn.Conv2d(C2, C3, kernel_size=3, padding=1, stride=2)
        self.block3 = ConvBlock(C3, C3)
        self.down3 = nn.Conv2d(C3, C3, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        s1 = self.block1(F.relu(self.stem(x)))
        s2 = self.block2(F.relu(self.down1(s1)))
        s3 = self.block3(F.relu(self.down2(s2)))
        bottom = F.relu(self.down3(s3))
        return s1, s2, s3, bottom

def adjacency(edges, edge_mask, n=O.MAX_NODES):
    """Constructs an adjacency matrix of all node neighbours"""
    B, E, _ = edges.shape
    adj = torch.zeros(B, n, n, device=edges.device)
    ex = torch.arange(B, device=edges.device).unsqueeze(1).expand(B, E)
    frm = edges[:, :, 0].long()
    to = edges[:, :, 1].long()
    keep = edge_mask.bool()
    adj[ex[keep], to[keep], frm[keep]] = 1.0
    adj[ex[keep], frm[keep], to[keep]] = 1.0
    return adj

def neighbour_mean(h, adj):
    """Finds the average of all the neighbours of a node"""
    total = adj @ h
    count = adj.sum(dim=2, keepdim=True)
    return total / count.clamp(min=1)

def masked_mean(h, mask):
    """Finds the average of all real nodes' rows"""
    m = mask.float().unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

def head_rows(h, nodes):
    """Adds 1 to the head node (of the queue) and 0 to all other ones"""
    w = nodes[:, :, O.G_HEAD].unsqueeze(-1)
    return (h * w).sum(dim=1)

class GraphEncoder(nn.Module):
    """Embeds the graph info into linear layers, combining info about neighbours in layer passes"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(O.N_NODE, GRAPH_DIM)
        self.mix1 = nn.Linear(2 * GRAPH_DIM, GRAPH_DIM)
        self.mix2 = nn.Linear(2 * GRAPH_DIM, GRAPH_DIM)
        self.out = nn.Linear(2 * GRAPH_DIM, GRAPH_DIM)

    def forward(self, nodes, node_mask, edges, edge_rate, edge_mask):
        adj = adjacency(edges, edge_mask)
        h = F.relu(self.embed(nodes))
        m = neighbour_mean(h, adj)
        h = F.relu(self.mix1(torch.cat([h, m], dim=-1)))
        m = neighbour_mean(h, adj)
        h = F.relu(self.mix2(torch.cat([h, m], dim=-1)))
        g = torch.cat([masked_mean(h, node_mask), head_rows(h, nodes)], dim=-1)
        return F.relu(self.out(g))
 
class Summary(nn.Module):
    """Creates a summary of all information, along with a predicted score and scores for all N positions"""
 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SUMMARY_IN, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.value = nn.Linear(HID, 1)
        self.n_head = nn.Linear(HID, N_MAX + 1)
 
    def forward(self, bottom, graph, vec, stage):
        pooled = bottom.mean(dim=(2, 3))
        glued = torch.cat([pooled, graph, vec, stage], dim=-1)
        z = F.relu(self.fc2(F.relu(self.fc1(glued))))
        value = self.value(z).squeeze(-1)
        n_logits = self.n_head(z)
        return z, value, n_logits

def grow(x):
    """Doubles a map's size by copying each cell into a 2 x 2 patch"""
    return F.interpolate(x, scale_factor=2, mode="nearest")
 
def add_to_every_cell(x, v):
    """Adds the same list v to every cell of map"""
    return x + v[:, :, None, None]
 
class Decoder(nn.Module):
    """Grows the 6 x 6 bottom back to 48 x 48 and gives one score per (rotation, floor, 2 m square)"""
 
    def __init__(self):
        super().__init__()
        self.inject = nn.Linear(HID, C3)
        self.up3 = ConvBlock(C3 + C3, C3)
        self.up2 = ConvBlock(C3 + C2, C2)
        self.up1 = ConvBlock(C2 + C1, C1)
        self.head = nn.Conv2d(32, N_MAPS, kernel_size=1)
 
    def forward(self, s1, s2, s3, bottom, z):
        x = add_to_every_cell(bottom, self.inject(z))
        x = grow(x)
        x = self.up3(torch.cat([x, s3], dim=1))
        x = grow(x)
        x = self.up2(torch.cat([x, s2], dim=1))
        x = grow(x)
        x = self.up1(torch.cat([x, s1], dim=1))
        x = self.head(x)
        return x.reshape(x.shape[0], -1)

def stack_obs(obs_list):
    """Stacks a list of observations into one dict"""
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
 
def obs_to_torch(obs, device="cpu"):
    """Converts batched numpy arrays to tensors, made to work on the provided device"""
    out = {}
    for k, v in obs.items():
        t = torch.as_tensor(v, device=device)
        if k == "edges":
            t = t.long()
        elif k in ("node_mask", "edge_mask"):
            t = t.bool()
        elif k != "grid":
            t = t.float()
        out[k] = t
    return out
 
def merge_logits(n_logits, pos_logits, stage):
    """Create a single score list per example (choosing from scores for N or position based on stage)"""
    k = n_logits.shape[1]
    n_full = torch.cat([n_logits, pos_logits[:, k:]], dim=1)
    choosing_n = (stage[:, 0] < 0.5).unsqueeze(1)
    return torch.where(choosing_n, n_full, pos_logits)
 
class Policy(nn.Module):
    """The whole network for the model"""
 
    def __init__(self):
        super().__init__()
        self.grid = GridEncoder()
        self.graph = GraphEncoder()
        self.summary = Summary()
        self.decoder = Decoder()
 
    def forward(self, obs):
        s1, s2, s3, bottom = self.grid(flatten_floors(obs["grid"]))
        g = self.graph(obs["nodes"], obs["node_mask"], obs["edges"], obs["edge_rate"], obs["edge_mask"])
        z, value, n_logits = self.summary(bottom, g, obs["vec"], obs["stage"])
        pos_logits = self.decoder(s1, s2, s3, bottom, z)
        logits = merge_logits(n_logits, pos_logits, obs["stage"])
        return logits, value
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Policy().to(device)
    print(f"{sum(p.numel() for p in net.parameters()):,} learned numbers, running on {device}")
 
    env = LayoutEnv()
    env.reset(random_task(np.random.default_rng(0)))
    obs = obs_to_torch(stack_obs([observe(env, 0, 0)] * 64), device)
    with torch.no_grad():
        for _ in range(3):
            net(obs)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            net(obs)
        if device == "cuda":
            torch.cuda.synchronize()
    print(f"one forward pass for 64 situations: {1000 * (time.perf_counter() - t) / 20:.1f} ms")
 
if __name__ == "__main__":
    main()