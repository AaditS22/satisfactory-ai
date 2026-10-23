import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from layout.baseline import baseline_a
from layout.clone import clone_loss, run_epoch, main
from layout.gym_env import N_POS
from layout.policy import Policy
from layout.teacher import teacher_task, GRID_SHAPE
from rl.checkpoint import load_checkpoint


class FakePolicy(nn.Module):
    def __init__(self, logits, value):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, obs):
        return self.logits, self.value


def run(logits, value, mask, actions, scores, vf_coef=0.5):
    policy = FakePolicy(logits, value)
    masks = torch.tensor(mask, dtype=torch.bool)
    return policy, clone_loss(policy, {}, masks, torch.tensor(actions), torch.tensor(scores), vf_coef)


def test_hand_numbers():
    _, (total, pl, vl, acc) = run([[0.0, math.log(3.0), 0.0, 0.0]], [0.2], [[True] * 4], [1], [0.7])
    assert abs(pl.item() - math.log(2.0)) < 1e-5
    assert abs(vl.item() - 0.25) < 1e-6
    assert abs(total.item() - (math.log(2.0) + 0.5 * 0.25)) < 1e-5
    assert acc == 1.0


def test_mask_removes_illegal_moves():
    _, (total, pl, vl, acc) = run([[0.0, math.log(3.0), 0.0, 0.0]], [0.7], [[False, True, False, True]], [1], [0.7])
    assert abs(pl.item() + math.log(0.75)) < 1e-5
    assert abs(vl.item()) < 1e-9


def test_losses_are_means_over_batch():
    _, (total, pl, vl, acc) = run([[0.0, 0.0], [0.0, 0.0]], [0.0, 1.0], [[True, True], [True, True]], [0, 1], [0.0, 0.0])
    assert abs(pl.item() - math.log(2.0)) < 1e-5
    assert abs(vl.item() - 0.5) < 1e-6


def test_accuracy_counts_argmax_hits():
    _, (total, pl, vl, acc) = run([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]], [0.0] * 4,
                                  [[True, True]] * 4, [0, 1, 1, 1], [0.0] * 4)
    assert abs(acc - 0.75) < 1e-9 and isinstance(acc, float)


def test_accuracy_ignores_masked_argmax():
    _, (total, pl, vl, acc) = run([[5.0, 1.0, 0.0]], [0.0], [[False, True, True]], [1], [0.0])
    assert acc == 1.0


def test_vf_coef_scales_value_part():
    _, (total, pl, vl, acc) = run([[0.0, 0.0]], [0.0], [[True, True]], [0], [1.0], vf_coef=2.0)
    assert abs(total.item() - (math.log(2.0) + 2.0)) < 1e-5


def test_gradient_step_raises_teacher_move():
    policy, (total, pl, vl, acc) = run([[1.0, 0.0, 0.0]], [0.0], [[True] * 3], [2], [0.5])
    total.backward()
    with torch.no_grad():
        p_before = torch.softmax(policy.logits, dim=1)[0, 2].item()
        for w in policy.parameters():
            w -= 0.5 * w.grad
        p_after = torch.softmax(policy.logits, dim=1)[0, 2].item()
    assert p_after > p_before
    assert policy.value.item() > 0.0


def test_returns_scalars():
    _, (total, pl, vl, acc) = run([[0.0, 0.0]] * 3, [0.1, 0.2, 0.3], [[True, True]] * 3, [0, 1, 0], [0.5] * 3)
    assert total.dim() == 0 and pl.dim() == 0 and vl.dim() == 0


class FlatPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_POS))
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        B = obs["vec"].shape[0]
        return self.logits.expand(B, N_POS), self.value.expand(B)


def fake_samples(n, action=7, score=0.6):
    return {"grid_bits": np.zeros((n, int(np.prod(GRID_SHAPE)) // 8), dtype=np.uint8),
            "mask_bits": np.packbits(np.ones((n, N_POS), dtype=bool), axis=1),
            "vec": np.zeros((n, 23), dtype=np.float32), "stage": np.zeros((n, 2), dtype=np.float32),
            "nodes": np.zeros((n, 40, 15), dtype=np.float32), "node_mask": np.zeros((n, 40), dtype=bool),
            "edges": np.zeros((n, 64, 2), dtype=np.int32), "edge_rate": np.zeros((n, 64), dtype=np.float32),
            "edge_mask": np.zeros((n, 64), dtype=bool),
            "action": np.full(n, action, dtype=np.int64), "score": np.full(n, score, dtype=np.float32)}


def test_run_epoch_validation_keeps_weights_and_hand_loss():
    policy = FlatPolicy()
    before = [w.detach().clone() for w in policy.parameters()]
    res = run_epoch(policy, fake_samples(10), 4, "cpu")
    for a, b in zip(before, policy.parameters()):
        assert torch.equal(a, b)
    assert sorted(res) == ["accuracy", "loss", "policy_loss", "value_loss"]
    assert abs(res["policy_loss"] - math.log(N_POS)) < 1e-3
    assert abs(res["value_loss"] - 0.36) < 1e-5
    assert abs(res["loss"] - (math.log(N_POS) + 0.5 * 0.36)) < 1e-3


def test_run_epoch_training_lowers_loss():
    policy = FlatPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=0.1)
    rng = np.random.default_rng(0)
    first = run_epoch(policy, fake_samples(12), 4, "cpu", opt, rng)
    for _ in range(5):
        last = run_epoch(policy, fake_samples(12), 4, "cpu", opt, rng)
    after = run_epoch(policy, fake_samples(12), 4, "cpu")
    assert after["policy_loss"] < first["policy_loss"] - 1.0
    assert after["value_loss"] < first["value_loss"]
    assert after["accuracy"] == 1.0


def test_main_saves_best_clone():
    rows = []
    for s in range(6):
        task = teacher_task(1, s)
        a = baseline_a(task)
        t = dict(task)
        t["alternates"] = sorted(t["alternates"])
        rows.append({"seed": s, "task": t, "plan": a["plan"], "score": float(a["score"])})
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "teacher.json"
        data.write_text(json.dumps(rows))
        out = Path(d) / "clone.pt"
        old = sys.argv
        sys.argv = ["clone", "--data", str(data), "--out", str(out), "--epochs", "2", "--minibatch", "4", "--device", "cpu"]
        try:
            main()
        finally:
            sys.argv = old
        assert out.exists()
        epoch, extra = load_checkpoint(out, Policy())
        assert epoch in (0, 1) and extra["stage"] == 1 and extra["val_loss"] > 0


TESTS = [test_hand_numbers, test_mask_removes_illegal_moves, test_losses_are_means_over_batch, test_accuracy_counts_argmax_hits,
    test_accuracy_ignores_masked_argmax, test_vf_coef_scales_value_part, test_gradient_step_raises_teacher_move, test_returns_scalars,
    test_run_epoch_validation_keeps_weights_and_hand_loss, test_run_epoch_training_lowers_loss, test_main_saves_best_clone]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")