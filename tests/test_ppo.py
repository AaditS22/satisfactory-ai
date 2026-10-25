import math

import numpy as np
import torch
from torch import nn

from rl.buffer import RolloutBuffer
from rl.dist import masked_log_probs, pick
from rl.ppo import ppo_losses, normalise, scale_adv, to_tensors, update, linear_lr


def one(ratio, adv, clip=0.2):
    new_logp = torch.tensor([math.log(ratio)], requires_grad=True)
    old_logp = torch.zeros(1)
    pl, vl = ppo_losses(new_logp, old_logp, torch.tensor([adv]), torch.zeros(1), torch.zeros(1), clip)
    return pl, new_logp


def test_returns_scalars():
    pl, vl = ppo_losses(torch.zeros(4), torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4))
    assert pl.dim() == 0 and vl.dim() == 0


def test_ratio_one_is_minus_mean_adv():
    logp = torch.tensor([-1.0, -2.0, -0.5])
    pl, _ = ppo_losses(logp, logp.clone(), torch.tensor([0.2, -0.2, 0.5]), torch.zeros(3), torch.zeros(3))
    assert abs(pl.item() + 0.5 / 3) < 1e-6


def test_lesson_table():
    for ratio, adv, want in [(1.1, 0.2, -0.22), (1.33, 0.2, -0.24), (0.7, -0.2, 0.16), (1.3, -0.2, 0.26),
            (0.7, 0.2, -0.14)]:
        pl, _ = one(ratio, adv)
        assert abs(pl.item() - want) < 1e-5, (ratio, adv, pl.item())


def test_ratio_uses_old_logp():
    pl, _ = ppo_losses(torch.tensor([-2.0]), torch.tensor([-2.0 - math.log(1.1)]), torch.tensor([0.2]),
        torch.zeros(1), torch.zeros(1))
    assert abs(pl.item() + 0.22) < 1e-5


def test_clip_parameter():
    pl, _ = one(1.15, 0.2, clip=0.1)
    assert abs(pl.item() + 0.22) < 1e-5
    pl, _ = one(1.15, 0.2, clip=0.2)
    assert abs(pl.item() + 0.23) < 1e-5


def test_gradient_free_zone():
    pl, new_logp = one(1.1, 0.2)
    pl.backward()
    assert abs(new_logp.grad.item() + 0.22) < 1e-5


def test_gradient_zero_when_braked():
    for ratio, adv in [(1.33, 0.2), (0.7, -0.2)]:
        pl, new_logp = one(ratio, adv)
        pl.backward()
        assert new_logp.grad.item() == 0.0, (ratio, adv)


def test_gradient_wrong_way_pushes_back():
    pl, new_logp = one(0.7, 0.2)
    pl.backward()
    assert abs(new_logp.grad.item() + 0.14) < 1e-5
    pl, new_logp = one(1.3, -0.2)
    pl.backward()
    assert abs(new_logp.grad.item() - 0.26) < 1e-5


def test_value_loss_hand():
    new_value = torch.tensor([0.3, 0.6], requires_grad=True)
    _, vl = ppo_losses(torch.zeros(2), torch.zeros(2), torch.zeros(2), new_value, torch.tensor([0.5, 0.5]))
    assert abs(vl.item() - 0.025) < 1e-6
    vl.backward()
    assert torch.allclose(new_value.grad, torch.tensor([-0.2, 0.1]))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = nn.Linear(3, 4)
        self.v = nn.Linear(3, 1)

    def forward(self, obs):
        x = obs["x"]
        return self.pi(x), self.v(x).squeeze(1)


class CountingAdam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def tiny_to_torch(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def make_buffer(policy, T=4, N=2, seed=0, same_x=False, actions=None):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(T, N, 4)
    for t in range(T):
        x = np.ones((N, 3), np.float32) if same_x else rng.normal(size=(N, 3)).astype(np.float32)
        masks = rng.random((N, 4)) < 0.6
        masks[:, 0] = True
        masks[:, 1] = True
        a = actions[t] if actions is not None else np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        with torch.no_grad():
            logits, value = policy({"x": torch.as_tensor(x)})
            logp = pick(masked_log_probs(logits, torch.as_tensor(masks)), torch.as_tensor(a))
        buf.add({"x": x}, masks, a, logp.numpy(), value.numpy(), rng.random(N).astype(np.float32),
            rng.random(N) < 0.3)
    buf.finish(np.zeros(N, np.float32))
    return buf


def test_normalise_hand():
    assert torch.allclose(normalise(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)


def test_to_tensors_types():
    d = to_tensors({"masks": np.ones((2, 3), bool), "actions": np.array([1, 2]), "adv": np.array([0.5, 1.0])}, "cpu")
    assert d["masks"].dtype == torch.bool and d["actions"].dtype == torch.int64 and d["adv"].dtype == torch.float32


def test_update_returns_stats():
    torch.manual_seed(0)
    policy = Tiny()
    buf = make_buffer(policy)
    stats = update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=2, minibatch=3,
        rng=np.random.default_rng(0))
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "push"}
    assert all(math.isfinite(v) for v in stats.values())
    assert stats["entropy"] > 0


def test_update_step_count():
    torch.manual_seed(0)
    policy = Tiny()
    buf = make_buffer(policy)
    opt = CountingAdam(policy.parameters(), lr=1e-3)
    update(policy, opt, buf, tiny_to_torch, epochs=2, minibatch=3, rng=np.random.default_rng(0))
    assert opt.steps == 6


def test_update_first_group_has_ratio_one():
    torch.manual_seed(0)
    policy = Tiny()
    buf = make_buffer(policy)
    stats = update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=1, minibatch=8,
        rng=np.random.default_rng(0))
    assert abs(stats["approx_kl"]) < 1e-6 and stats["clip_frac"] == 0.0


def test_update_changes_every_weight():
    torch.manual_seed(0)
    policy = Tiny()
    buf = make_buffer(policy)
    before = [p.detach().clone() for p in policy.parameters()]
    update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=1, minibatch=8,
        rng=np.random.default_rng(0))
    for b, p in zip(before, policy.parameters()):
        assert not torch.equal(b, p.detach())


def test_update_vf_coef_zero_leaves_value_head():
    torch.manual_seed(0)
    policy = Tiny()
    buf = make_buffer(policy)
    before = policy.v.weight.detach().clone()
    update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=1, minibatch=8,
        vf_coef=0.0, rng=np.random.default_rng(0))
    assert torch.equal(before, policy.v.weight.detach())


def test_update_pushes_good_action_up():
    torch.manual_seed(0)
    policy = Tiny()
    actions = [np.array([0, 1])] * 8
    buf = make_buffer(policy, T=8, same_x=True, actions=actions)
    buf.adv = np.where(buf.actions == 0, 1.0, -1.0).astype(np.float32)
    x = {"x": torch.ones(1, 3)}
    with torch.no_grad():
        p_before = torch.softmax(policy(x)[0], 1)[0]
    update(policy, torch.optim.Adam(policy.parameters(), lr=5e-2), buf, tiny_to_torch, epochs=4, minibatch=16,
        ent_coef=0.0, rng=np.random.default_rng(0))
    with torch.no_grad():
        p_after = torch.softmax(policy(x)[0], 1)[0]
    assert p_after[0] > p_before[0] and p_after[1] < p_before[1]


def test_scale_adv_hand():
    assert torch.allclose(scale_adv(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(scale_adv(torch.tensor([0.002, -0.002]), 0.02), torch.tensor([0.1, -0.1]))
    late = torch.tensor([0.001, 0.0, -0.001])
    assert torch.allclose(scale_adv(late), torch.tensor([1.0, 0.0, -1.0]), atol=1e-4)
    assert torch.allclose(scale_adv(late, 0.03), torch.tensor([1 / 30, 0.0, -1 / 30]), atol=1e-6)


def test_scale_adv_fixed_keeps_mean():
    assert torch.allclose(scale_adv(torch.full((3,), 0.05), 0.05), torch.ones(3))
    assert torch.allclose(scale_adv(torch.full((3,), 0.05)), torch.zeros(3))


def even_buffer(policy, size):
    buf = make_buffer(policy)
    buf.adv = np.where(np.arange(8).reshape(4, 2) % 2 == 0, size, -size).astype(np.float32)
    return buf


def test_update_push_fixed_vs_curve():
    for scale, want in [(0.02, 0.5), (None, math.sqrt(7 / 8))]:
        torch.manual_seed(0)
        policy = Tiny()
        buf = even_buffer(policy, 0.01)
        stats = update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=1,
            minibatch=8, rng=np.random.default_rng(0), adv_scale=scale)
        assert abs(stats["push"] - want) < 1e-4, (scale, stats["push"])


def test_update_push_fixed_scale_minibatches():
    torch.manual_seed(0)
    policy = Tiny()
    buf = even_buffer(policy, 0.01)
    stats = update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), buf, tiny_to_torch, epochs=2, minibatch=3,
        rng=np.random.default_rng(0), adv_scale=0.02)
    assert abs(stats["push"] - 0.5) < 1e-6


def test_update_fixed_scale_pushes_good_action_up():
    torch.manual_seed(0)
    policy = Tiny()
    actions = [np.array([0, 1])] * 8
    buf = make_buffer(policy, T=8, same_x=True, actions=actions)
    buf.adv = np.where(buf.actions == 0, 0.02, -0.02).astype(np.float32)
    x = {"x": torch.ones(1, 3)}
    with torch.no_grad():
        p_before = torch.softmax(policy(x)[0], 1)[0]
    update(policy, torch.optim.Adam(policy.parameters(), lr=5e-2), buf, tiny_to_torch, epochs=4, minibatch=16,
        ent_coef=0.0, rng=np.random.default_rng(0), adv_scale=0.02)
    with torch.no_grad():
        p_after = torch.softmax(policy(x)[0], 1)[0]
    assert p_after[0] > p_before[0] and p_after[1] < p_before[1]


def test_linear_lr_hand():
    assert math.isclose(linear_lr(0, 1000, 2.5e-4), 2.5e-4)
    assert math.isclose(linear_lr(500, 1000, 2.5e-4), 1.25e-4)
    assert math.isclose(linear_lr(750, 1000, 2.5e-4), 6.25e-5)
    assert math.isclose(linear_lr(999, 1000, 2.5e-4), 2.5e-7)
    assert math.isclose(linear_lr(2, 4, 1.0), 0.5)


def test_linear_lr_follows_rounds():
    for rounds in (1, 7, 300, 5000):
        lrs = [linear_lr(r, rounds, 1e-3) for r in range(rounds)]
        assert math.isclose(lrs[0], 1e-3)
        assert all(x > 0 for x in lrs)
        assert all(a > b for a, b in zip(lrs, lrs[1:]))
        assert math.isclose(lrs[-1], 1e-3 / rounds)


TESTS = [test_returns_scalars, test_ratio_one_is_minus_mean_adv, test_lesson_table, test_ratio_uses_old_logp,
    test_clip_parameter, test_gradient_free_zone, test_gradient_zero_when_braked, test_gradient_wrong_way_pushes_back,
    test_value_loss_hand, test_normalise_hand, test_to_tensors_types, test_update_returns_stats, test_update_step_count,
    test_update_first_group_has_ratio_one, test_update_changes_every_weight, test_update_vf_coef_zero_leaves_value_head,
    test_update_pushes_good_action_up, test_scale_adv_hand, test_scale_adv_fixed_keeps_mean,
    test_update_push_fixed_vs_curve, test_update_push_fixed_scale_minibatches,
    test_update_fixed_scale_pushes_good_action_up, test_linear_lr_hand, test_linear_lr_follows_rounds]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")