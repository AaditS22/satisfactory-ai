import numpy as np

from rl.buffer import gae, RolloutBuffer, loo_advantage, fix_broken, EpisodeBuffer


def f32(x):
    return np.array(x, dtype=np.float32)


def lesson_episode():
    rewards = f32([[0.0], [0.0], [0.5]])
    values = f32([[0.30], [0.60], [0.55]])
    dones = np.array([[False], [False], [True]])
    return rewards, values, dones, f32([9.0])


def test_gae_lesson_example():
    adv, ret = gae(*lesson_episode(), gamma=1.0, lam=0.5)
    assert np.allclose(adv[:, 0], [0.2625, -0.075, -0.05])
    assert np.allclose(ret[:, 0], [0.5625, 0.525, 0.5])


def test_gae_lam_one_is_final_minus_prediction():
    adv, ret = gae(*lesson_episode(), gamma=1.0, lam=1.0)
    assert np.allclose(adv[:, 0], [0.2, -0.1, -0.05])
    assert np.allclose(ret[:, 0], [0.5, 0.5, 0.5])


def test_gae_lam_zero_is_own_jump():
    adv, _ = gae(*lesson_episode(), gamma=1.0, lam=0.0)
    assert np.allclose(adv[:, 0], [0.3, -0.05, -0.05])


def test_gae_default_lam():
    adv, _ = gae(*lesson_episode())
    assert np.allclose(adv[:, 0], [0.207375, -0.0975, -0.05])


def test_gae_shapes_dtypes_inputs_unchanged():
    r, v, d, last = lesson_episode()
    v_before = v.copy()
    adv, ret = gae(r, v, d, last)
    assert adv.shape == (3, 1) and ret.shape == (3, 1)
    assert adv.dtype == np.float32 and ret.dtype == np.float32
    assert np.array_equal(v, v_before)


def test_gae_done_stops_borrowing():
    rewards = f32([[0.0], [0.8], [0.0], [0.0]])
    values = f32([[0.5], [0.6], [0.2], [0.3]])
    dones = np.array([[False], [True], [False], [False]])
    adv, ret = gae(rewards, values, dones, f32([0.4]), gamma=1.0, lam=1.0)
    assert np.allclose(adv[:, 0], [0.3, 0.2, 0.2, 0.1])
    assert np.allclose(ret[:, 0], [0.8, 0.8, 0.4, 0.4])


def test_gae_uses_last_value_only_if_not_done():
    rewards = f32([[0.0, 1.0]])
    values = f32([[0.2, 0.3]])
    dones = np.array([[False, True]])
    adv, _ = gae(rewards, values, dones, f32([0.7, 99.0]))
    assert np.allclose(adv[0], [0.5, 0.7])


def test_gae_gamma():
    rewards = f32([[1.0], [1.0]])
    values = f32([[0.0], [0.0]])
    dones = np.array([[False], [True]])
    adv, _ = gae(rewards, values, dones, f32([0.0]), gamma=0.5, lam=1.0)
    assert np.allclose(adv[:, 0], [1.5, 1.0])


def test_gae_columns_independent():
    rng = np.random.default_rng(0)
    rewards = rng.random((6, 4)).astype(np.float32)
    values = rng.random((6, 4)).astype(np.float32)
    dones = rng.random((6, 4)) < 0.3
    last = rng.random(4).astype(np.float32)
    adv, ret = gae(rewards, values, dones, last, gamma=0.9, lam=0.8)
    for n in range(4):
        a1, r1 = gae(rewards[:, n:n + 1], values[:, n:n + 1], dones[:, n:n + 1], last[n:n + 1], gamma=0.9, lam=0.8)
        assert np.allclose(adv[:, n], a1[:, 0]) and np.allclose(ret[:, n], r1[:, 0])


def filled_buffer(T=3, N=2, A=5):
    buf = RolloutBuffer(T, N, A)
    for t in range(T):
        obs = {"grid": np.full((N, 2, 2), t, dtype=np.uint8), "vec": np.arange(N, dtype=np.float32) + 10 * t}
        masks = np.zeros((N, A), dtype=bool)
        masks[:, t] = True
        buf.add(obs, masks, np.arange(N) + t, f32([-0.5] * N), f32([0.1 * t] * N), f32([1.0 * (t == T - 1)] * N),
            np.array([t == T - 1] * N))
    return buf


def test_buffer_stores_rows():
    buf = filled_buffer()
    assert buf.t == 3
    assert buf.obs["grid"].shape == (3, 2, 2, 2) and buf.obs["grid"].dtype == np.uint8
    assert buf.masks.shape == (3, 2, 5) and buf.masks[1, 0, 1] and not buf.masks[1, 0, 2]
    assert np.array_equal(buf.actions, [[0, 1], [1, 2], [2, 3]])
    assert np.array_equal(buf.obs["vec"], [[0, 1], [10, 11], [20, 21]])


def test_buffer_full_raises():
    buf = filled_buffer()
    try:
        buf.add({"grid": np.zeros((2, 2, 2), np.uint8), "vec": f32([0, 0])}, np.zeros((2, 5), bool),
            np.zeros(2, int), f32([0, 0]), f32([0, 0]), f32([0, 0]), np.zeros(2, bool))
    except ValueError:
        return
    assert False


def test_buffer_finish_needs_full():
    buf = RolloutBuffer(3, 2, 5)
    try:
        buf.finish(f32([0, 0]))
    except ValueError:
        return
    assert False


def test_buffer_finish_matches_gae():
    buf = filled_buffer()
    buf.finish(f32([0.7, 0.7]), gamma=1.0, lam=0.5)
    adv, ret = gae(buf.rewards, buf.values, buf.dones, f32([0.7, 0.7]), 1.0, 0.5)
    assert np.allclose(buf.adv, adv) and np.allclose(buf.ret, ret)
    assert np.allclose(buf.adv[:, 0], [0.35, 0.5, 0.8])


def test_buffer_get_flat_order():
    buf = filled_buffer()
    buf.finish(f32([0, 0]))
    obs, data = buf.get(np.array([3, 0, 5]))
    assert np.array_equal(obs["vec"], [11, 0, 21])
    assert np.array_equal(data["actions"], [2, 0, 3])
    assert obs["grid"].shape == (3, 2, 2) and obs["grid"][0, 0, 0] == 1
    assert data["masks"].shape == (3, 5) and data["masks"][0, 1] and data["masks"][2, 2]
    assert np.allclose(data["values"], [0.1, 0.0, 0.2])
    assert np.allclose(data["adv"], [buf.adv[1, 1], buf.adv[0, 0], buf.adv[2, 1]])
    assert np.allclose(data["ret"], [buf.ret[1, 1], buf.ret[0, 0], buf.ret[2, 1]])
    assert np.allclose(data["logp"], [-0.5, -0.5, -0.5])


def test_buffer_get_is_a_copy():
    buf = filled_buffer()
    buf.finish(f32([0, 0]))
    obs, data = buf.get(np.arange(6))
    obs["vec"][:] = -1
    data["adv"][:] = -1
    assert buf.obs["vec"][0, 1] == 1 and buf.adv[0, 0] != -1


def test_buffer_reset_refills():
    buf = filled_buffer()
    buf.reset()
    assert buf.t == 0
    buf.add({"grid": np.full((2, 2, 2), 9, np.uint8), "vec": f32([5, 6])}, np.ones((2, 5), bool),
        np.array([4, 4]), f32([0, 0]), f32([0, 0]), f32([0, 0]), np.zeros(2, bool))
    assert buf.t == 1 and buf.obs["grid"][0, 0, 0, 0] == 9 and np.array_equal(buf.actions[0], [4, 4])


def test_loo_lesson_group():
    r = np.array([[0.74, 0.76, 0.75, 0.75, 0.77, 0.73, 0.76, 0.74]])
    adv = loo_advantage(r)
    assert adv.shape == (1, 8) and adv.dtype == np.float32
    assert np.isclose(adv[0, 0], 0.74 - 5.26 / 7, atol=1e-6)
    assert np.isclose(adv[0, 4], 0.77 - 5.23 / 7, atol=1e-6)


def test_loo_small_hand():
    adv = loo_advantage(np.array([[1.0, 2.0, 6.0], [0.5, 0.5, 0.5]]))
    assert np.allclose(adv, [[-3.0, -1.5, 4.5], [0.0, 0.0, 0.0]])


def test_loo_rows_are_groups_even_when_square():
    r = np.array([[0.0, 1.0], [10.0, 30.0]])
    assert np.allclose(loo_advantage(r), [[-1.0, 1.0], [-20.0, 20.0]])


def test_loo_sums_to_zero_and_scales_mean_gap():
    r = np.random.default_rng(0).normal(0.7, 0.05, size=(5, 8))
    adv = loo_advantage(r)
    assert np.allclose(adv.sum(axis=1), 0.0, atol=1e-5)
    assert np.allclose(adv, 8 / 7 * (r - r.mean(axis=1, keepdims=True)), atol=1e-6)


def test_loo_input_unchanged_and_needs_two():
    r = np.array([[0.2, 0.4]])
    loo_advantage(r)
    assert np.array_equal(r, [[0.2, 0.4]])
    try:
        loo_advantage(np.array([[0.5], [0.6]]))
    except ValueError:
        return
    raise AssertionError("one copy per group must raise ValueError")


def test_fix_lesson_group():
    r = np.array([[0.6, 0.5, -0.7, 0.7]])
    out = fix_broken(r, np.array([[False, False, True, False]]), 0.1)
    assert np.allclose(out, [[0.6, 0.5, 0.4, 0.7]])
    assert np.allclose(loo_advantage(out)[0, [0, 2]], [0.6 - 1.6 / 3, 0.4 - 0.6], atol=1e-6)


def test_fix_every_broken_copy_gets_the_same_floor():
    r = np.array([[0.7, -1.5, 0.8, 0.9]])
    out = fix_broken(r, np.array([[False, True, False, True]]), 0.05)
    assert np.allclose(out, [[0.7, 0.65, 0.8, 0.65]])


def test_fix_rows_use_their_own_worst():
    r = np.array([[0.2, -0.5, 0.3], [0.9, 0.8, -0.4]])
    b = np.array([[False, True, False], [False, False, True]])
    assert np.allclose(fix_broken(r, b, 0.1), [[0.2, 0.1, 0.3], [0.9, 0.8, 0.7]])


def test_fix_clean_and_all_broken_rows():
    r = np.array([[0.5, 0.6], [-0.7, -1.2]])
    out = fix_broken(r, np.array([[False, False], [True, True]]), 0.1)
    assert np.allclose(out[0], [0.5, 0.6])
    assert np.isclose(out[1, 0], out[1, 1])
    assert np.allclose(loo_advantage(out)[1], 0.0)


def test_fix_input_unchanged():
    r = np.array([[0.5, -0.7]])
    b = np.array([[False, True]])
    fix_broken(r, b, 0.1)
    assert np.array_equal(r, [[0.5, -0.7]]) and np.array_equal(b, [[False, True]])


def step_obs(i):
    return {"x": np.array([float(i)], dtype=np.float32)}


def put(buf, slot, i):
    buf.add(slot, step_obs(i), np.array([True, i % 2 == 0]), i % 2, -0.5, 0.25)


def hand_page():
    buf = EpisodeBuffer(copies=2)
    put(buf, 0, 10)
    put(buf, 1, 20)
    put(buf, 0, 11)
    put(buf, 1, 21)
    put(buf, 2, 30)
    put(buf, 1, 22)
    buf.end_game(1, 7, 0.6, False)
    buf.end_game(2, 9, 0.5, False)
    buf.end_game(0, 7, 0.8, False)
    put(buf, 0, 40)
    buf.end_game(0, 9, 0.5, False)
    buf.finish()
    return buf


def test_episode_hand_page():
    buf = hand_page()
    assert buf.T == 7 and buf.N == 1
    obs, d = buf.get(np.arange(7))
    assert np.array_equal(obs["x"][:, 0], [20, 21, 22, 30, 10, 11, 40])
    assert np.allclose(d["adv"], [-0.2, -0.2, -0.2, 0.0, 0.2, 0.2, 0.0])
    assert np.allclose(d["ret"], [0.6, 0.6, 0.6, 0.5, 0.8, 0.8, 0.5])
    assert np.array_equal(d["actions"], [0, 1, 0, 0, 0, 1, 0]) and np.allclose(d["logp"], -0.5)
    assert np.array_equal(d["masks"][:, 1], [True, False, True, True, True, False, True])


def test_episode_get_like_rollout_buffer():
    buf = hand_page()
    obs, d = buf.get(np.array([3, 0]))
    assert set(d) == {"masks", "actions", "logp", "values", "adv", "ret"}
    assert np.array_equal(obs["x"][:, 0], [30, 20])
    assert d["masks"].dtype == bool and d["actions"].dtype == np.int64
    assert all(d[k].dtype == np.float32 for k in ("logp", "values", "adv", "ret"))


def test_episode_stats():
    st = hand_page().stats
    assert st["steps"] == 7 and st["groups"] == 2
    assert np.isclose(st["flat"], 0.5) and np.isclose(st["adv"], 0.1)


def test_episode_broken_game_graded_from_floor_but_value_target_raw():
    buf = EpisodeBuffer(copies=2, margin=0.1)
    put(buf, 0, 1)
    put(buf, 1, 2)
    buf.end_game(0, 3, 0.6, False)
    buf.end_game(1, 3, -0.7, True)
    buf.finish()
    _, d = buf.get(np.arange(2))
    assert np.allclose(d["adv"], [0.1, -0.1]) and np.allclose(d["ret"], [0.6, -0.7])


def test_episode_finish_refuses_open_or_uneven_groups():
    buf = EpisodeBuffer(copies=2)
    put(buf, 0, 1)
    put(buf, 1, 2)
    put(buf, 2, 3)
    buf.end_game(0, 3, 0.6, False)
    buf.end_game(1, 3, 0.5, False)
    try:
        buf.finish()
        raise AssertionError("open game must raise")
    except ValueError:
        pass
    buf.end_game(2, 3, 0.4, False)
    put(buf, 0, 4)
    put(buf, 1, 5)
    put(buf, 2, 6)
    for slot in range(3):
        buf.end_game(slot, 8, 0.5, False)
    try:
        buf.finish()
        raise AssertionError("3 copies with copies=2 must raise")
    except ValueError:
        pass


def test_episode_reset_refills():
    buf = hand_page()
    buf.reset()
    assert buf.T == 0
    put(buf, 0, 1)
    put(buf, 1, 2)
    buf.end_game(0, 5, 0.3, False)
    buf.end_game(1, 5, 0.3, False)
    buf.finish()
    assert buf.T == 2 and np.allclose(buf.get(np.arange(2))[1]["adv"], 0.0)


TESTS = [test_gae_lesson_example, test_gae_lam_one_is_final_minus_prediction, test_gae_lam_zero_is_own_jump,
    test_gae_default_lam, test_gae_shapes_dtypes_inputs_unchanged, test_gae_done_stops_borrowing,
    test_gae_uses_last_value_only_if_not_done, test_gae_gamma, test_gae_columns_independent, test_buffer_stores_rows,
    test_buffer_full_raises, test_buffer_finish_needs_full, test_buffer_finish_matches_gae, test_buffer_get_flat_order,
    test_buffer_get_is_a_copy, test_buffer_reset_refills, test_loo_lesson_group, test_loo_small_hand,
    test_loo_rows_are_groups_even_when_square, test_loo_sums_to_zero_and_scales_mean_gap,
    test_loo_input_unchanged_and_needs_two, test_fix_lesson_group, test_fix_every_broken_copy_gets_the_same_floor,
    test_fix_rows_use_their_own_worst, test_fix_clean_and_all_broken_rows, test_fix_input_unchanged,
    test_episode_hand_page, test_episode_get_like_rollout_buffer, test_episode_stats,
    test_episode_broken_game_graded_from_floor_but_value_target_raw, test_episode_finish_refuses_open_or_uneven_groups,
    test_episode_reset_refills]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")