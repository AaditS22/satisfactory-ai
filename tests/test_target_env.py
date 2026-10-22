import numpy as np

from rl.target_env import PickTarget


def play(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rewards = []
    done = False
    while not done:
        a = policy(env, obs)
        obs, r, done, trunc, info = env.step(a)
        rewards.append(r)
    return rewards


def perfect(env, obs):
    return int(np.argmax(obs["target"]))


def test_reset_obs():
    env = PickTarget()
    obs, info = env.reset(seed=0)
    assert set(obs) == {"target", "progress"}
    assert obs["target"].shape == (8,) and obs["target"].dtype == np.float32
    assert obs["target"].sum() == 1.0
    assert obs["progress"].dtype == np.float32 and np.array_equal(obs["progress"], [0.0, 0.0])
    assert info == {}


def test_mask_has_target_and_n_legal():
    env = PickTarget(n_actions=10, n_legal=3)
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        m = env.action_masks()
        assert m.dtype == bool and m.shape == (10,) and m.sum() == 3
        assert m[np.argmax(obs["target"])]


def test_action_masks_is_a_copy():
    env = PickTarget()
    env.reset(seed=0)
    m = env.action_masks()
    m[:] = False
    assert env.action_masks().sum() == 4


def test_perfect_player_reward_only_at_end():
    env = PickTarget(n_steps=3)
    assert play(env, perfect, 0) == [0.0, 0.0, 1.0]


def test_partial_hits():
    env = PickTarget(n_steps=4)
    calls = []

    def miss_first(env, obs):
        calls.append(1)
        if len(calls) == 1:
            m = env.action_masks()
            m[np.argmax(obs["target"])] = False
            return int(np.flatnonzero(m)[0])
        return perfect(env, obs)

    assert play(env, miss_first, 1) == [0.0, 0.0, 0.0, 0.75]


def test_progress_counts_steps_and_hits():
    env = PickTarget(n_steps=4)
    obs, _ = env.reset(seed=2)
    m = env.action_masks()
    m[np.argmax(obs["target"])] = False
    obs, r, done, trunc, info = env.step(int(np.flatnonzero(m)[0]))
    assert np.allclose(obs["progress"], [0.25, 0.0])
    obs, r, done, trunc, info = env.step(perfect(env, obs))
    assert np.allclose(obs["progress"], [0.5, 0.25]) and info["hits"] == 1


def test_done_state():
    env = PickTarget(n_steps=2)
    env.reset(seed=3)
    obs, _, _, _, _ = env.step(perfect(env, env._obs()))
    obs, r, done, trunc, info = env.step(perfect(env, obs))
    assert done and not trunc and r == 1.0
    assert env.action_masks().sum() == 0 and obs["target"].sum() == 0
    try:
        env.step(0)
        assert False
    except ValueError:
        pass


def test_illegal_action_raises():
    env = PickTarget()
    env.reset(seed=4)
    bad = int(np.flatnonzero(~env.action_masks())[0])
    for a in [bad, -1, 8]:
        try:
            env.step(a)
            assert False, a
        except ValueError:
            pass


def test_seed_repeats():
    a, b = PickTarget(), PickTarget()
    for seed in [0, 7]:
        oa, _ = a.reset(seed=seed)
        ob, _ = b.reset(seed=seed)
        assert np.array_equal(oa["target"], ob["target"]) and np.array_equal(a.action_masks(), b.action_masks())


def test_random_player_scores_one_over_n_legal():
    env = PickTarget()
    rng = np.random.default_rng(0)
    total = 0.0
    for i in range(4000):
        total += play(env, lambda e, o: rng.choice(np.flatnonzero(e.action_masks())), i)[-1]
    assert abs(total / 4000 - 0.25) < 0.01


TESTS = [test_reset_obs, test_mask_has_target_and_n_legal, test_action_masks_is_a_copy,
    test_perfect_player_reward_only_at_end, test_partial_hits, test_progress_counts_steps_and_hits, test_done_state,
    test_illegal_action_raises, test_seed_repeats, test_random_player_scores_one_over_n_legal]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")