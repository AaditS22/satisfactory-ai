import numpy as np
import torch
from torch import nn

from rl.buffer import RolloutBuffer
from rl.collect import stack, act, collect
from rl.games import LocalGames, ProcGames
from rl.target_env import PickTarget


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = nn.Linear(10, 8)
        self.v = nn.Linear(10, 1)
        self.grad_on = None

    def forward(self, obs):
        self.grad_on = torch.is_grad_enabled()
        x = torch.cat([obs["target"], obs["progress"]], 1)
        return self.pi(x), self.v(x).squeeze(1)


def to_torch(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def boards(n, seed=0):
    envs = [PickTarget() for _ in range(n)]
    obs = [e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]
    masks = np.stack([e.action_masks() for e in envs])
    return envs, obs, masks


def test_stack_shapes_and_order():
    out = stack([{"a": np.array([1, 2]), "b": np.float32(5)}, {"a": np.array([3, 4]), "b": np.float32(6)}])
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (2, 2) and np.array_equal(out["a"], [[1, 2], [3, 4]])
    assert out["b"].shape == (2,) and out["b"].dtype == np.float32


def test_act_shapes_dtypes():
    torch.manual_seed(0)
    _, obs, masks = boards(5)
    actions, logp, values = act(Tiny(), obs, masks, to_torch)
    for a in (actions, logp, values):
        assert isinstance(a, np.ndarray) and a.shape == (5,)
    assert actions.dtype == np.int64 and logp.dtype == np.float32 and values.dtype == np.float32


def test_act_no_grad():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(2)
    act(policy, obs, masks, to_torch)
    assert policy.grad_on is False


def test_act_only_legal():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(6)
    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        actions, _, _ = act(policy, obs, masks, to_torch, gen)
        assert masks[np.arange(6), actions].all()


def test_act_logp_and_values_match_network():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(4)
    actions, logp, values = act(policy, obs, masks, to_torch, torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, v = policy(to_torch(stack(obs), "cpu"))
        log_p = torch.log_softmax(torch.where(torch.as_tensor(masks), logits, -1e9), 1)
    want = log_p[torch.arange(4), torch.as_tensor(actions)].numpy()
    assert np.allclose(logp, want, atol=1e-5)
    assert np.allclose(values, v.numpy(), atol=1e-6)


def test_act_gen_repeats():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(6)
    a1, _, _ = act(policy, obs, masks, to_torch, torch.Generator().manual_seed(3))
    a2, _, _ = act(policy, obs, masks, to_torch, torch.Generator().manual_seed(3))
    assert np.array_equal(a1, a2)


def test_act_one_legal():
    torch.manual_seed(0)
    _, obs, _ = boards(3)
    masks = np.zeros((3, 8), dtype=bool)
    masks[0, 6] = True
    masks[1, 0] = True
    masks[2, 3] = True
    actions, logp, _ = act(Tiny(), obs, masks, to_torch)
    assert np.array_equal(actions, [6, 0, 3])
    assert np.abs(logp).max() < 1e-6


def page(T, N, seed=0, policy=None):
    torch.manual_seed(seed)
    policy = policy or Tiny()
    games = LocalGames(PickTarget, N)
    obs, masks = games.reset(seed)
    buf = RolloutBuffer(T, N, 8)
    obs, masks, finished = collect(games, obs, masks, policy, buf, to_torch, torch.Generator().manual_seed(seed))
    return policy, obs, masks, finished, buf


def test_collect_fills_and_finishes():
    _, obs, masks, finished, buf = page(5, 3)
    assert buf.t == 5 and buf.adv.shape == (5, 3) and buf.ret.shape == (5, 3)
    assert isinstance(obs, list) and len(obs) == 3 and set(obs[0]) == {"target", "progress"}
    assert masks.shape == (3, 8) and masks.dtype == bool
    assert isinstance(finished, list)


def test_collect_dones_every_third_tick():
    _, _, _, _, buf = page(6, 4)
    for i in range(4):
        assert buf.dones[:, i].tolist() == [False, False, True, False, False, True]


def test_collect_stores_obs_before_move():
    _, _, _, _, buf = page(6, 2)
    for i in range(2):
        assert np.allclose(buf.obs["progress"][:, i, 0], [0, 1 / 3, 2 / 3, 0, 1 / 3, 2 / 3], atol=1e-6)
        assert np.allclose(buf.obs["target"][:, i].sum(1), 1.0)


def test_collect_stored_masks_match_stored_obs():
    _, _, _, _, buf = page(6, 3)
    targets = buf.obs["target"].argmax(2)
    t, n = np.meshgrid(np.arange(6), np.arange(3), indexing="ij")
    assert buf.masks[t, n, targets].all() and (buf.masks.sum(2) == 4).all()


def test_collect_rewards_only_at_game_end():
    _, _, _, _, buf = page(6, 5)
    assert (buf.rewards[~buf.dones] == 0).all()
    for r in buf.rewards[buf.dones]:
        assert min(abs(r - k / 3) for k in range(4)) < 1e-6


def test_collect_finished_infos():
    _, _, _, finished, buf = page(6, 5)
    assert len(finished) == 10
    assert all("score" in f and "hits" in f for f in finished)
    assert np.allclose(sorted(f["score"] for f in finished), sorted(buf.rewards[buf.dones].tolist()))
    assert all(abs(f["score"] - f["hits"] / 3) < 1e-6 for f in finished)


def test_collect_actions_legal_and_logp_finite():
    _, _, _, _, buf = page(6, 5)
    t, n = np.meshgrid(np.arange(6), np.arange(5), indexing="ij")
    assert buf.masks[t, n, buf.actions].all()
    assert np.isfinite(buf.logp).all() and (buf.logp <= 0).all()


def test_collect_continues_games_across_pages():
    torch.manual_seed(0)
    policy = Tiny()
    games = LocalGames(PickTarget, 3)
    obs, masks = games.reset(0)
    buf = RolloutBuffer(2, 3, 8)
    gen = torch.Generator().manual_seed(0)
    obs, masks, _ = collect(games, obs, masks, policy, buf, to_torch, gen)
    assert np.allclose([o["progress"][0] for o in obs], 2 / 3)
    obs, masks, finished = collect(games, obs, masks, policy, buf, to_torch, gen)
    assert np.allclose(buf.obs["progress"][0, :, 0], 2 / 3, atol=1e-6)
    assert buf.dones[0].all() and not buf.dones[1].any() and len(finished) == 3


def test_collect_last_value_used_for_unfinished_games():
    policy, obs, _, _, buf = page(4, 3)
    with torch.no_grad():
        v_last = policy(to_torch(stack(obs), "cpu"))[1].numpy()
    assert not buf.dones[3].any()
    assert np.allclose(buf.adv[3], v_last - buf.values[3], atol=1e-5)


def test_collect_with_worker_processes_matches_local():
    _, _, _, f1, b1 = page(6, 4, seed=3)
    torch.manual_seed(3)
    policy = Tiny()
    with ProcGames(PickTarget, 4, 2) as games:
        obs, masks = games.reset(3)
        buf = RolloutBuffer(6, 4, 8)
        obs, masks, f2 = collect(games, obs, masks, policy, buf, to_torch, torch.Generator().manual_seed(3))
    assert np.array_equal(b1.actions, buf.actions) and np.array_equal(b1.rewards, buf.rewards)
    assert f1 == f2

TESTS = [test_stack_shapes_and_order, test_act_shapes_dtypes, test_act_no_grad, test_act_only_legal,
    test_act_logp_and_values_match_network, test_act_gen_repeats, test_act_one_legal, test_collect_fills_and_finishes,
    test_collect_dones_every_third_tick, test_collect_stores_obs_before_move, test_collect_stored_masks_match_stored_obs,
    test_collect_rewards_only_at_game_end, test_collect_finished_infos, test_collect_actions_legal_and_logp_finite,
    test_collect_continues_games_across_pages, test_collect_last_value_used_for_unfinished_games,
    test_collect_with_worker_processes_matches_local]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")