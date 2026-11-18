from pathlib import Path

import torch

def save_checkpoint(path, policy, optimizer, round, extra=None):
    """Save everything needed to use the network later or to continue training it"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "round": round, 
        "extra": extra or {}
    }
    torch.save(save, path)

def strip_checkpoint(path, out_path, info=None):
    """Save only the policy weights of a training checkpoint (no optimizer state), for sharing.
    The result loads with load_checkpoint (without an optimizer) and with torch.load(weights_only=True)."""
    data = torch.load(path, map_location="cpu")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": data["policy"], "round": data["round"], "extra": info or {}}, out_path)

def load_checkpoint(path, policy, optimizer=None, device="cpu"):
    """Put a saved checkpoint's numbers back into an existing policy and optimizer"""
    data = torch.load(path, map_location=device)
    policy.load_state_dict(data["policy"])
    if optimizer:
        optimizer.load_state_dict(data["optimizer"])
    return data["round"], data["extra"]