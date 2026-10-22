import numpy as np
import torch

from rl.collect import stack, act
from rl.target_env import PickTarget
from rl.train_target import TargetNet, obs_to_torch, HIDDEN, train


def batch(n, seed=0):
    envs = [PickTarget() for _ in range(n)]
    obs = [e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]
    masks = np.stack([e.action_masks() for e in envs])
    return obs, masks


def test_obs_to_torch_float32():
    obs, _ = batch(3)
    x = obs_to_torch(stack(obs))
    assert x["target"].dtype == torch.float32 and x["target"].shape == (3, 8)
    assert x["progress"].dtype == torch.float32 and x["progress"].shape == (3, 2)


def test_net_shapes():
    torch.manual_seed(0)
    obs, _ = batch(3)
    logits, value = TargetNet()(obs_to_torch(stack(obs)))
    assert logits.shape == (3, 8) and value.shape == (3,)


def test_net_other_action_count():
    torch.manual_seed(0)
    x = {"target": torch.zeros(2, 5), "progress": torch.zeros(2, 2)}
    logits, value = TargetNet(n_actions=5)(x)
    assert logits.shape == (2, 5) and value.shape == (2,)


def test_net_param_count():
    assert HIDDEN == 64
    assert sum(p.numel() for p in TargetNet().parameters()) == 5449
    names = {n for n, _ in TargetNet().named_parameters()}
    assert names == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "pi.weight", "pi.bias", "v.weight", "v.bias"}


def test_net_reads_both_inputs():
    torch.manual_seed(0)
    net = TargetNet()
    a = {"target": torch.zeros(1, 8), "progress": torch.zeros(1, 2)}
    b = {"target": torch.zeros(1, 8), "progress": torch.zeros(1, 2)}
    a["target"][0, 1] = 1.0
    b["target"][0, 6] = 1.0
    assert not torch.allclose(net(a)[0], net(b)[0])
    c = {"target": a["target"], "progress": torch.tensor([[0.0, 0.0]])}
    d = {"target": a["target"], "progress": torch.tensor([[2 / 3, 2 / 3]])}
    assert not torch.allclose(net(c)[1], net(d)[1])


def test_net_value_loss_reaches_body_not_policy_head():
    torch.manual_seed(0)
    net = TargetNet()
    x = {"target": torch.eye(8)[:4], "progress": torch.rand(4, 2)}
    _, value = net(x)
    value.sum().backward()
    assert net.fc1.weight.grad is not None and net.fc1.weight.grad.abs().sum() > 0
    assert net.pi.weight.grad is None or net.pi.weight.grad.abs().sum() == 0


def test_net_logits_can_be_negative():
    torch.manual_seed(0)
    x = {"target": torch.eye(8), "progress": torch.rand(8, 2)}
    logits, value = TargetNet()(x)
    assert (logits < 0).any()


def test_net_works_with_act():
    torch.manual_seed(0)
    obs, masks = batch(6)
    actions, logp, values = act(TargetNet(), obs, masks, obs_to_torch, torch.Generator().manual_seed(0))
    assert masks[np.arange(6), actions].all() and values.shape == (6,)


def weights(net):
    return [p.detach().clone() for p in net.parameters()]


def test_train_short_run():
    policy, history = train(rounds=3, n_games=4, T=6, minibatch=8, verbose=False)
    assert isinstance(policy, TargetNet)
    assert isinstance(history, list) and len(history) == 3
    assert all(isinstance(h, float) and 0.0 <= h <= 1.0 for h in history)


def test_train_changes_weights():
    torch.manual_seed(0)
    fresh = weights(TargetNet())
    policy, _ = train(rounds=2, n_games=4, T=6, minibatch=8, seed=0, verbose=False)
    assert any(not torch.equal(a, b) for a, b in zip(fresh, weights(policy)))


def test_train_same_seed_same_run():
    p1, h1 = train(rounds=3, n_games=4, T=6, minibatch=8, seed=5, verbose=False)
    p2, h2 = train(rounds=3, n_games=4, T=6, minibatch=8, seed=5, verbose=False)
    assert h1 == h2
    assert all(torch.equal(a, b) for a, b in zip(weights(p1), weights(p2)))


def test_train_different_seed_differs():
    p1, _ = train(rounds=2, n_games=4, T=6, minibatch=8, seed=1, verbose=False)
    p2, _ = train(rounds=2, n_games=4, T=6, minibatch=8, seed=2, verbose=False)
    assert any(not torch.equal(a, b) for a, b in zip(weights(p1), weights(p2)))


TESTS = [test_obs_to_torch_float32, test_net_shapes, test_net_other_action_count, test_net_param_count,
    test_net_reads_both_inputs, test_net_value_loss_reaches_body_not_policy_head, test_net_logits_can_be_negative,
    test_net_works_with_act, test_train_short_run, test_train_changes_weights, test_train_same_seed_same_run,
    test_train_different_seed_differs]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")