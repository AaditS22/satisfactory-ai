import argparse

import numpy as np
import torch

from layout.gym_env import StageEnv
from layout.policy import Policy, obs_to_torch
from layout.best_of import play_jobs
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.collect import stack
from rl.dist import masked_log_probs, sample, entropy
from rl.games import ProcGames

FIRST_JOB = 900_000_000  # far away from every training seed (1,000,000 x (seed + 1) + ...)


def soften(policy, T):
    """Divide every output logit of the policy by T, in place, by scaling its two output layers.

    policy: a Policy. T: float > 0 (T > 1 makes sampling less sure, T = 1 changes nothing).
    The greedy move never changes, because dividing all logits by the same T keeps their order.
    Returns nothing.
    """
    with torch.no_grad():
        for layer in (policy.summary.n_head, policy.decoder.head):
            layer.weight.div_(T)
            layer.bias.div_(T)


def load_soft(path, T, device):
    """A fresh Policy with a checkpoint's weights, softened by T"""
    policy = Policy().to(device)
    load_checkpoint(path, policy, device=device)
    soften(policy, T)
    policy.eval()
    return policy


def entropy_chooser(policy, gen, record):
    """A choose(obs_list, masks) for play_jobs that samples moves and appends each step's entropy to record"""
    device = next(policy.parameters()).device

    def choose(obs_list, masks):
        x = obs_to_torch(stack(obs_list), device)
        m = torch.as_tensor(masks, device=device)
        with torch.no_grad():
            logits, _ = policy(x)
            log_p = masked_log_probs(logits, m)
            record.extend(entropy(log_p).cpu().tolist())
            return sample(log_p, gen).cpu().numpy()

    return choose


def measure(path, temps, stage, games, slots, workers, seed, device):
    """Mean entropy per step when the softened policy plays `games` training tasks, for every T in temps"""
    out = {}
    jobs = list(range(FIRST_JOB, FIRST_JOB + games))
    with ProcGames(StageEnv(stage, solvable=True), slots, workers) as env:
        for T in temps:
            policy = load_soft(path, T, device)
            gen = torch.Generator(device=device).manual_seed(seed)
            record = []
            play_jobs(env, jobs, entropy_chooser(policy, gen, record))
            out[T] = float(np.mean(record))
    return out


def pick_T(res, target):
    """Smallest T whose entropy reaches target, or the largest T tried if none does"""
    temps = sorted(res)
    for T in temps:
        if res[T] >= target:
            return T
    return temps[-1]


def save_soft(path, T, out_path, device):
    """Save the softened weights as a checkpoint that --init can load"""
    policy = load_soft(path, T, device)
    optimizer = torch.optim.Adam(policy.parameters())
    save_checkpoint(out_path, policy, optimizer, 0, {"softened_from": path, "T": T})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stage", type=int, default=3)
    p.add_argument("--temps", type=float, nargs="+", default=[1.0])
    p.add_argument("--games", type=int, default=224)
    p.add_argument("--save", nargs="+", default=None, help="where to write softened checkpoints (one per --target, or one with a single --temps value)")
    p.add_argument("--target", type=float, nargs="+", default=None, help="entropies to reach: picks the smallest T for each and saves it")
    p.add_argument("--slots", type=int, default=48)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.target and (not args.save or len(args.save) != len(args.target)):
        p.error("--target needs one --save path per target")
    if args.save and not args.target:
        if len(args.temps) != 1 or len(args.save) != 1:
            p.error("--save without --target needs exactly one --temps value and one path")
        save_soft(args.checkpoint, args.temps[0], args.save[0], device)
        print(f"saved {args.save[0]} (T {args.temps[0]})")
        return

    res = measure(args.checkpoint, args.temps, args.stage, args.games, args.slots, args.workers, args.seed, device)
    print(f"{args.games} stage-{args.stage} training tasks, sampled moves")
    for T, e in res.items():
        print(f"T {T:5.2f}   entropy {e:.3f}")
    for target, path in zip(args.target or [], args.save or []):
        T = pick_T(res, target)
        save_soft(args.checkpoint, T, path, device)
        note = "" if res[T] >= target else "  (target NOT reached, used the largest T)"
        print(f"target {target}: T {T} (entropy {res[T]:.3f}) saved {path}{note}")


if __name__ == "__main__":
    main()