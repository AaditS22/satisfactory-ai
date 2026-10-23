import torch

from rl.dist import masked_log_probs, pick

import argparse
import math
import time
import numpy as np

from layout.policy import Policy, obs_to_torch
from layout.teacher import load_teacher, split_rows, build, batch
from rl.checkpoint import save_checkpoint, load_checkpoint

def clone_loss(policy, obs, masks, actions, scores, vf_coef=0.5):
    """Finds the loss for one minibatch on behaviour-cloning learning"""
    logits, value = policy(obs)
    log_probs = masked_log_probs(logits, masks)
    pick_probs = pick(log_probs, actions)

    policy_loss = -pick_probs.mean()
    value_loss = (value - scores).square().mean()
    total = policy_loss + vf_coef * value_loss
    accuracy = (log_probs.argmax(dim=1) == actions).float().mean().item()

    return total, policy_loss, value_loss, accuracy

def to_torch(obs, masks, actions, scores, device):
    """Turns one unpacked minibatch to tensors"""
    return (obs_to_torch(obs, device), torch.as_tensor(masks, device=device),
            torch.as_tensor(actions, device=device), torch.as_tensor(scores, device=device))

def run_epoch(policy, samples, minibatch, device, optimizer=None, rng=None, max_grad_norm=0.5):
    """Does one pass over samples, minibatch rows at a time, training optimizer or measuring on validation set"""
    n = len(samples["action"])
    order = rng.permutation(n) if optimizer is not None else np.arange(n)
    sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}
    for start in range(0, n, minibatch):
        idx = order[start:start + minibatch]
        obs, masks, actions, scores = to_torch(*batch(samples, idx), device)
        if optimizer is None:
            with torch.no_grad():
                total, pl, vl, acc = clone_loss(policy, obs, masks, actions, scores)
        else:
            total, pl, vl, acc = clone_loss(policy, obs, masks, actions, scores)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
        k = len(idx)
        sums["loss"] += total.item() * k
        sums["policy_loss"] += pl.item() * k
        sums["value_loss"] += vl.item() * k
        sums["accuracy"] += acc * k
    return {key: v / n for key, v in sums.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--data", default=None, help="default: data/teacher_stage<N>.json")
    p.add_argument("--out", default=None, help="default: runs/clone_stage<N>/clone.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--init", default=None, help="start from this checkpoint's weights")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = args.data or f"data/teacher_stage{args.stage}.json"
    out = args.out or f"runs/clone_stage{args.stage}/clone.pt"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    train_rows, val_rows = split_rows(load_teacher(data))
    train, val = build(train_rows), build(val_rows)
    print(f"{len(train_rows)} train tasks ({len(train['action'])} steps), "
          f"{len(val_rows)} val tasks ({len(val['action'])} steps), built in {time.time() - t0:.0f} s")

    policy = Policy().to(device)
    if args.init:
        load_checkpoint(args.init, policy, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best = math.inf
    for epoch in range(args.epochs):
        t = time.time()
        tr = run_epoch(policy, train, args.minibatch, device, optimizer, rng)
        va = run_epoch(policy, val, args.minibatch, device)
        print(f"epoch {epoch:2d}  train loss {tr['loss']:.3f} acc {tr['accuracy']:.2f}  |  val loss {va['loss']:.3f} "
              f"(policy {va['policy_loss']:.3f}, value {va['value_loss']:.4f}) acc {va['accuracy']:.2f}  {time.time() - t:.0f} s")
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(out, policy, optimizer, epoch, {"val_loss": best, "stage": args.stage})
    print(f"best val loss {best:.3f}, saved {out}")

if __name__ == "__main__":
    main()