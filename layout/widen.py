import argparse

import torch

from layout.policy import Policy
from rl.checkpoint import load_checkpoint, save_checkpoint

def widen(policy, src=8):
    """Copies the n head's row for n = src into every row for a bigger n"""
    w = policy.summary.n_head.weight
    b = policy.summary.n_head.bias

    with torch.no_grad():
        for n in range(src + 1, 17):
            w[n] = w[src]
            b[n] = b[src]

def save_wide(path, src, out_path, device="cpu"):
    """Save a checkpoint's weights with the n head widened as a checkpoint that --init can load"""
    policy = Policy().to(device)
    load_checkpoint(path, policy, device=device)
    widen(policy, src)
    optimizer = torch.optim.Adam(policy.parameters())
    save_checkpoint(out_path, policy, optimizer, 0, {"widened_from": path, "src": src})
    return policy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--src", type=int, default=8)
    p.add_argument("--save", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    policy = save_wide(args.checkpoint, args.src, args.save, args.device)
    rows = policy.summary.n_head.weight.shape[0]
    print(f"saved {args.save}: n head rows {args.src + 1}-{rows - 1} now equal row {args.src}")

if __name__ == "__main__":
    main()