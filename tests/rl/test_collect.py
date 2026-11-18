import numpy as np
import torch
from torch import nn

from rl.buffer import RolloutBuffer, EpisodeBuffer
import time

from rl.collect import stack, act, collect, Background, collect_groups, collect_groups_async
from rl.dist import masked_log_probs, pick
from rl.ppo import update
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


def test_act_greedy_picks_most_likely_legal():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(6)
    actions, logp, _ = act(policy, obs, masks, to_torch, greedy=True)
    with torch.no_grad():
        logits, _ = policy(to_torch(stack(obs), "cpu"))
        log_p = torch.log_softmax(torch.where(torch.as_tensor(masks), logits, -1e9), 1)
    assert np.array_equal(actions, log_p.argmax(1).numpy())
    assert masks[np.arange(6), actions].all()
    assert np.allclose(logp, log_p.max(1).values.numpy(), atol=1e-6)
    assert actions.dtype == np.int64


def test_act_greedy_ignores_randomness():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(6)
    a1, _, _ = act(policy, obs, masks, to_torch, torch.Generator().manual_seed(1), greedy=True)
    a2, _, _ = act(policy, obs, masks, to_torch, torch.Generator().manual_seed(2), greedy=True)
    assert np.array_equal(a1, a2)


def test_act_default_still_samples():
    torch.manual_seed(0)
    policy = Tiny()
    with torch.no_grad():
        policy.pi.weight.zero_()
        policy.pi.bias.zero_()
    _, obs, masks = boards(1)
    picks = {int(act(policy, obs, masks, to_torch, torch.Generator().manual_seed(s))[0][0]) for s in range(40)}
    assert len(picks) == 4

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


def bg_for(policy, n, seed, T=6):
    games = LocalGames(PickTarget, n)
    obs, masks = games.reset(seed)
    bufs = [RolloutBuffer(T, n, 8), RolloutBuffer(T, n, 8)]
    return Background(policy, games, bufs, to_torch, torch.Generator().manual_seed(seed)), bufs, obs, masks


def test_background_page_matches_collect():
    _, _, masks1, f1, b1 = page(6, 4, seed=2)
    torch.manual_seed(2)
    policy = Tiny()
    bg, bufs, obs, masks = bg_for(policy, 4, 2)
    bg.sync(policy)
    bg.start(obs, masks)
    buf, obs, masks, f2, secs = bg.wait()
    assert buf is bufs[0]
    assert np.array_equal(b1.actions, buf.actions) and np.array_equal(b1.rewards, buf.rewards)
    assert np.allclose(b1.logp, buf.logp) and np.allclose(b1.adv, buf.adv)
    assert f1 == f2 and np.array_equal(masks, masks1) and secs > 0


def test_background_reuses_one_thread():
    # a new thread per page leaks native memory when the actor runs on CUDA, so every page must use the same thread
    torch.manual_seed(2)
    policy = Tiny()
    bg, _, obs, masks = bg_for(policy, 4, 2)
    seen = set()
    for _ in range(5):
        bg.start(obs, masks)
        seen.add(bg.thread.ident)
        _, obs, masks, _, _ = bg.wait()
    worker = bg._worker
    assert len(seen) == 1 and worker.is_alive()
    bg.close()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_background_actor_is_a_copy():
    _, _, _, _, b1 = page(6, 4, seed=2)
    torch.manual_seed(2)
    policy = Tiny()
    bg, _, obs, masks = bg_for(policy, 4, 2)
    bg.sync(policy)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(5.0)
    bg.start(obs, masks)
    buf = bg.wait()[0]
    assert np.array_equal(buf.actions, b1.actions)
    assert not any(p is q for p, q in zip(policy.parameters(), bg.actor.parameters()))


def test_background_sync_copies_weights():
    torch.manual_seed(0)
    policy = Tiny()
    bg, _, _, _ = bg_for(policy, 2, 0)
    with torch.no_grad():
        policy.pi.bias.fill_(3.0)
    assert not torch.equal(bg.actor.pi.bias, policy.pi.bias)
    bg.sync(policy)
    assert torch.equal(bg.actor.pi.bias, policy.pi.bias) and bg.actor.pi.bias is not policy.pi.bias


def test_background_buffers_take_turns():
    torch.manual_seed(0)
    policy = Tiny()
    bg, bufs, obs, masks = bg_for(policy, 3, 0, T=2)
    bg.start(obs, masks)
    first, obs, masks, _, _ = bg.wait()
    kept = first.actions.copy()
    bg.start(obs, masks)
    second, obs, masks, _, _ = bg.wait()
    assert first is bufs[0] and second is bufs[1]
    assert np.array_equal(bufs[0].actions, kept)
    assert np.allclose(bufs[1].obs["progress"][0, :, 0], 2 / 3, atol=1e-6)
    bg.start(obs, masks)
    assert bg.wait()[0] is bufs[0]


class Broken:
    def step(self, actions):
        raise KeyError("boom")


def test_background_reraises_errors():
    torch.manual_seed(0)
    policy = Tiny()
    _, obs, masks = boards(2)
    bg = Background(policy, Broken(), [RolloutBuffer(2, 2, 8), RolloutBuffer(2, 2, 8)], to_torch)
    bg.start(obs, masks)
    try:
        bg.wait()
        assert False
    except KeyError:
        pass
    assert bg.thread is None


def test_background_one_page_at_a_time():
    torch.manual_seed(0)
    policy = Tiny()
    bg, _, obs, masks = bg_for(policy, 2, 0)
    bg.start(obs, masks)
    for f in (lambda: bg.sync(policy), lambda: bg.start(obs, masks)):
        try:
            f()
            assert False
        except RuntimeError:
            pass
    bg.wait()
    bg.sync(policy)

GROUP_JOBS = [s for s in (100, 101, 102) for _ in range(4)]


def group_page(games, policy, broken=lambda info: False):
    buf = EpisodeBuffer(copies=4)
    finished, idle = collect_groups(games, policy, buf, GROUP_JOBS, to_torch, broken, torch.Generator().manual_seed(0))
    return buf, finished, idle


def test_groups_play_every_job_once():
    torch.manual_seed(0)
    buf, finished, idle = group_page(LocalGames(PickTarget, 4), Tiny())
    assert len(finished) == 12 and buf.T == 36 and buf.N == 1
    assert buf.stats["groups"] == 3 and np.isclose(idle, 0.1)
    assert all("score" in f for f in finished)


def test_groups_every_step_gets_its_game_grade():
    torch.manual_seed(0)
    buf, _, _ = group_page(LocalGames(PickTarget, 4), Tiny())
    _, d = buf.get(np.arange(buf.T))
    for g in range(12):
        assert len(set(d["adv"][3 * g:3 * g + 3])) == 1 and len(set(d["ret"][3 * g:3 * g + 3])) == 1
    assert abs(float(d["adv"].sum())) < 1e-5
    assert set(np.round(d["ret"] * 3).astype(int)) <= {0, 1, 2, 3}


def test_groups_logp_and_values_match_network():
    torch.manual_seed(0)
    policy = Tiny()
    buf, _, _ = group_page(LocalGames(PickTarget, 4), policy)
    obs, d = buf.get(np.arange(buf.T))
    assert d["masks"][np.arange(buf.T), d["actions"]].all()
    with torch.no_grad():
        logits, value = policy(to_torch(obs, "cpu"))
        logp = pick(masked_log_probs(logits, torch.as_tensor(d["masks"])), torch.as_tensor(d["actions"]))
    assert np.allclose(logp.numpy(), d["logp"], atol=1e-5) and np.allclose(value.numpy(), d["values"], atol=1e-5)


def test_groups_broken_games_never_graded_up():
    torch.manual_seed(0)
    buf, _, _ = group_page(LocalGames(PickTarget, 4), Tiny(), broken=lambda info: info["score"] < 0.5)
    _, d = buf.get(np.arange(buf.T))
    assert (d["ret"] < 0.5).any() and (d["adv"][d["ret"] < 0.5] <= 1e-6).all()


def test_groups_proc_matches_local():
    torch.manual_seed(0)
    policy = Tiny()
    b1, f1, i1 = group_page(LocalGames(PickTarget, 4), policy)
    with ProcGames(PickTarget, 4, 2) as proc:
        b2, f2, i2 = group_page(proc, policy)
    _, d1 = b1.get(np.arange(b1.T))
    _, d2 = b2.get(np.arange(b2.T))
    assert all(np.array_equal(d1[k], d2[k]) for k in d1) and f1 == f2 and i1 == i2


def test_groups_page_feeds_update():
    torch.manual_seed(0)
    policy = Tiny()
    buf, _, _ = group_page(LocalGames(PickTarget, 4), policy)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = update(policy, opt, buf, to_torch, epochs=2, minibatch=8, rng=np.random.default_rng(0))
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "push"}
    assert all(np.isfinite(v) for v in stats.values())


class Sure(nn.Module):
    """Always plays the target (huge logit on it), so pages don't depend on the random draws"""
    def __init__(self):
        super().__init__()
        self.v = nn.Linear(10, 1)

    def forward(self, obs):
        x = torch.cat([obs["target"], obs["progress"]], 1)
        return 50.0 * obs["target"], self.v(x).squeeze(1)


class VarLen(PickTarget):
    """Game length depends on the seed (2, 3 or 4 moves); seeds ending in 7 are slow on their last move"""
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.K = 2 + seed % 3
            self.slow = seed % 10 == 7
        return super().reset(seed=seed, options=options)

    def step(self, action):
        out = super().step(action)
        if out[2] and getattr(self, "slow", False):
            time.sleep(0.05)
        return out


VAR_JOBS = [s for s in (100, 101, 102, 107, 117) for _ in range(3)]


def async_page(games, policy, jobs=VAR_JOBS, copies=3, broken=lambda info: False):
    buf = EpisodeBuffer(copies=copies)
    finished, idle = collect_groups_async(games, policy, buf, jobs, to_torch, broken, torch.Generator().manual_seed(0))
    return buf, finished, idle


def test_async_same_page_as_sync_when_moves_are_fixed():
    torch.manual_seed(0)
    policy = Sure()
    b1, f1, _ = group_page(LocalGames(PickTarget, 4), policy)
    b2, f2, _ = async_page(LocalGames(PickTarget, 4, n_parts=2), policy, GROUP_JOBS, 4)
    with ProcGames(PickTarget, 4, 2) as proc:
        b3, f3, _ = async_page(proc, policy, GROUP_JOBS, 4)
    assert b1.T == b2.T == b3.T == 36 and len(f1) == len(f2) == len(f3) == 12
    for b in (b2, b3):
        assert b.stats == b1.stats
        assert sorted(zip(b1.ret, b1.adv)) == sorted(zip(b.ret, b.adv))


def test_async_plays_every_job_with_out_of_step_workers():
    torch.manual_seed(0)
    policy = Tiny()
    with ProcGames(VarLen, 5, 3) as proc:
        buf, finished, idle = async_page(proc, policy)
    assert len(finished) == 15 and buf.stats["groups"] == 5
    assert buf.T == 3 * sum(2 + s % 3 for s in (100, 101, 102, 107, 117))
    assert 0 <= idle < 1
    obs, d = buf.get(np.arange(buf.T))
    assert d["masks"][np.arange(buf.T), d["actions"]].all()
    with torch.no_grad():
        logits, value = policy(to_torch(obs, "cpu"))
        logp = pick(masked_log_probs(logits, torch.as_tensor(d["masks"])), torch.as_tensor(d["actions"]))
    assert np.allclose(logp.numpy(), d["logp"], atol=1e-5) and np.allclose(value.numpy(), d["values"], atol=1e-5)


def test_async_each_game_is_whole_and_in_order():
    torch.manual_seed(0)
    buf, _, _ = async_page(LocalGames(VarLen, 4, n_parts=3), Tiny())
    obs, d = buf.get(np.arange(buf.T))
    starts = np.flatnonzero(obs["progress"][:, 0] == 0)
    assert len(starts) == 15
    for a, b in zip(starts, list(starts[1:]) + [buf.T]):
        game = obs["progress"][a:b, 0]
        assert (np.diff(game) > 0).all() and len(set(d["adv"][a:b])) == 1 and len(set(d["ret"][a:b])) == 1


def test_async_more_jobs_than_slots_and_one_part():
    torch.manual_seed(0)
    jobs = [s for s in range(200, 210) for _ in range(2)]
    buf, finished, _ = async_page(LocalGames(VarLen, 3), Tiny(), jobs, 2)
    assert len(finished) == 20 and buf.stats["groups"] == 10


class SlowEnd(VarLen):
    """Every game's last move takes 50 ms (like routing and scoring), games last 2-4 moves"""
    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.slow = True
        return out


def test_async_slow_steps_only_hold_up_their_own_worker():
    jobs = [s for s in range(300, 320) for _ in range(2)]
    policy = Sure()
    times = {}
    with ProcGames(SlowEnd, 4, 4) as proc:
        proc.reset(0)
        for name, play in (("sync", collect_groups), ("async", collect_groups_async)):
            buf = EpisodeBuffer(copies=2)
            t = time.perf_counter()
            f, _ = play(proc, policy, buf, jobs, to_torch, lambda info: False, torch.Generator().manual_seed(0))
            times[name] = time.perf_counter() - t
            assert len(f) == 40
    assert times["async"] < 0.8 * times["sync"], times


def test_background_groups_page():
    torch.manual_seed(0)
    policy = Sure()
    games = LocalGames(PickTarget, 4, n_parts=2)
    bufs = [EpisodeBuffer(4), EpisodeBuffer(4)]
    bg = Background(policy, games, bufs, to_torch, torch.Generator().manual_seed(0))
    bg.sync(policy)
    bg.start_groups(GROUP_JOBS, lambda info: False)
    buf, finished, idle, secs = bg.wait()
    assert buf is bufs[0] and len(finished) == 12 and buf.T == 36 and secs > 0 and 0 <= idle < 1
    bg.start_groups(GROUP_JOBS, lambda info: False, use_async=False)
    assert bg.wait()[0] is bufs[1]
