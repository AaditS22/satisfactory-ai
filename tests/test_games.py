import numpy as np

from rl.games import LocalGames, ProcGames, JobQueue
from rl.target_env import PickTarget


class TimeLimit(PickTarget):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return obs, r, False, term, info


def perfect(obs):
    return np.array([int(np.argmax(o["target"])) for o in obs])


def random_legal(masks, rng):
    return np.array([rng.choice(np.flatnonzero(m)) for m in masks])


def test_local_reset_shapes():
    obs, masks = LocalGames(PickTarget, 4).reset(seed=0)
    assert isinstance(obs, list) and len(obs) == 4 and set(obs[0]) == {"target", "progress"}
    assert masks.shape == (4, 8) and masks.dtype == bool and (masks.sum(1) == 4).all()


def test_local_reset_seeds():
    obs, masks = LocalGames(PickTarget, 3).reset(seed=10)
    for i in range(3):
        env = PickTarget()
        o, _ = env.reset(seed=10 + i)
        assert np.array_equal(obs[i]["target"], o["target"]) and np.array_equal(masks[i], env.action_masks())


def test_local_step_types():
    games = LocalGames(PickTarget, 3)
    obs, masks = games.reset(seed=0)
    obs, masks, rewards, dones, infos = games.step(perfect(obs))
    assert len(obs) == 3 and masks.shape == (3, 8) and masks.dtype == bool
    assert rewards.shape == (3,) and rewards.dtype == np.float32
    assert dones.shape == (3,) and dones.dtype == bool and not dones.any()
    assert isinstance(infos, list) and len(infos) == 3 and "score" not in infos[0]


def test_local_finished_games_restart():
    games = LocalGames(PickTarget, 3)
    obs, masks = games.reset(seed=0)
    for t in range(3):
        obs, masks, rewards, dones, infos = games.step(perfect(obs))
    assert dones.all() and np.allclose(rewards, 1.0)
    assert all(info["score"] == 1.0 for info in infos)
    assert all(np.array_equal(o["progress"], [0.0, 0.0]) for o in obs)
    assert (masks.sum(1) == 4).all() and masks[np.arange(3), perfect(obs)].all()


def test_local_truncated_counts_as_done():
    games = LocalGames(TimeLimit, 2)
    obs, masks = games.reset(seed=0)
    for t in range(3):
        obs, masks, rewards, dones, infos = games.step(perfect(obs))
    assert dones.all() and all(info["score"] == 1.0 for info in infos)


def test_local_masks_match_new_obs():
    games = LocalGames(PickTarget, 5)
    rng = np.random.default_rng(0)
    obs, masks = games.reset(seed=3)
    for t in range(7):
        obs, masks, _, _, _ = games.step(random_legal(masks, rng))
        assert masks[np.arange(5), perfect(obs)].all()


def test_proc_matches_local():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    local = LocalGames(PickTarget, 5)
    with ProcGames(PickTarget, 5, 2) as proc:
        o1, m1 = local.reset(seed=7)
        o2, m2 = proc.reset(seed=7)
        assert np.array_equal(m1, m2)
        for t in range(8):
            o1, m1, r1, d1, i1 = local.step(random_legal(m1, rng1))
            o2, m2, r2, d2, i2 = proc.step(random_legal(m2, rng2))
            assert np.array_equal(m1, m2) and np.array_equal(r1, r2) and np.array_equal(d1, d2)
            assert all(np.array_equal(a["target"], b["target"]) for a, b in zip(o1, o2))
            assert i1 == i2
            assert r2.dtype == np.float32 and d2.dtype == bool


def test_proc_more_workers_than_games():
    with ProcGames(PickTarget, 3, 8) as proc:
        assert len(proc.procs) == 3
        obs, masks = proc.reset(seed=0)
        obs, masks, rewards, dones, infos = proc.step(perfect(obs))
        assert len(obs) == 3 and masks.shape == (3, 8)


def test_proc_worker_error_is_raised():
    with ProcGames(PickTarget, 2, 2) as proc:
        obs, masks = proc.reset(seed=0)
        bad = np.array([int(np.flatnonzero(~m)[0]) for m in masks])
        try:
            proc.step(bad)
            assert False
        except RuntimeError as e:
            assert "illegal action" in str(e)


def test_proc_close_stops_workers():
    with ProcGames(PickTarget, 4, 2) as proc:
        proc.reset(seed=0)
        procs = list(proc.procs)
    assert not any(p.is_alive() for p in procs)


def fresh(seed):
    env = PickTarget()
    obs, _ = env.reset(seed=seed)
    return obs, env.action_masks()


def play_page(games, jobs, choose):
    queue = JobQueue(jobs)
    n = games.n
    playing, current, obs, masks = [False] * n, [None] * n, [None] * n, np.zeros((n, 8), bool)
    done_games, ticks = [], 0
    while len(queue) or any(playing):
        actions = [choose(o, m) if p else None for o, m, p in zip(obs, masks, playing)]
        offer = queue.offer(playing)
        obs, masks, rewards, dones, infos = games.step(actions, offer)
        for i in range(n):
            if dones[i]:
                done_games.append((current[i], infos[i]["score"]))
        for i, job in queue.taken(offer, playing, dones).items():
            current[i] = job
        playing = [o is not None for o in obs]
        ticks += 1
    return done_games, ticks


def test_jobs_free_slots_start_their_job():
    games = LocalGames(PickTarget, 3)
    obs, masks, rewards, dones, infos = games.step([None, None, None], [11, None, 11])
    o, m = fresh(11)
    assert np.array_equal(obs[0]["target"], o["target"]) and np.array_equal(obs[2]["target"], o["target"])
    assert np.array_equal(masks[0], m) and np.array_equal(masks[2], m)
    assert obs[1] is None and not masks[1].any()
    assert not dones.any() and np.allclose(rewards, 0.0)


def test_jobs_finished_game_takes_offer_or_goes_free():
    games = LocalGames(PickTarget, 2)
    obs, _, _, _, _ = games.step([None, None], [5, 6])
    for t in range(2):
        obs, _, _, dones, _ = games.step(perfect(obs), [None, None])
        assert not dones.any()
    obs, masks, rewards, dones, infos = games.step(perfect(obs), [None, 9])
    assert dones.all() and np.allclose(rewards, 1.0) and infos[0]["score"] == 1.0 and infos[1]["score"] == 1.0
    assert obs[0] is None and not masks[0].any()
    assert np.array_equal(obs[1]["target"], fresh(9)[0]["target"]) and np.array_equal(obs[1]["progress"], [0, 0])


def test_jobs_busy_slot_keeps_its_game():
    games = LocalGames(PickTarget, 1)
    obs, _, _, _, _ = games.step([None], [5])
    obs, _, _, dones, _ = games.step(perfect(obs), [7])
    assert not dones[0] and np.allclose(obs[0]["progress"], [1 / 3, 1 / 3])


def test_queue_offers_free_slots_first():
    assert JobQueue([1, 2, 3]).offer([True, False, True, False]) == [3, 1, None, 2]
    assert JobQueue([]).offer([False, True]) == [None, None]


def test_queue_removes_only_started_jobs():
    q = JobQueue([1, 2, 3])
    offer = q.offer([True, False, True, False])
    started = q.taken(offer, [True, False, True, False], [False, False, True, False])
    assert started == {1: 1, 3: 2} and len(q) == 1 and q.jobs == [3]


def test_page_plays_every_job_exactly_once():
    jobs = [s for s in (100, 101, 102, 103) for _ in range(3)]
    played, ticks = play_page(LocalGames(PickTarget, 5), jobs, lambda o, m: int(np.argmax(o["target"])))
    assert sorted(j for j, _ in played) == sorted(jobs)
    assert all(score == 1.0 for _, score in played)
    assert ticks == 1 + 3 * 3


def test_page_copies_with_same_moves_score_the_same():
    jobs = [s for s in (200, 201) for _ in range(4)]
    def first_legal(o, m):
        return int(np.flatnonzero(m)[0])
    played, _ = play_page(LocalGames(PickTarget, 3), jobs, first_legal)
    by_job = {}
    for j, score in played:
        by_job.setdefault(j, []).append(score)
    assert sorted(by_job) == [200, 201] and all(len(v) == 4 and len(set(v)) == 1 for v in by_job.values())


def test_proc_page_matches_local():
    jobs = [s for s in (300, 301, 302) for _ in range(4)]
    def chooser():
        rng = np.random.default_rng(4)
        return lambda o, m: int(rng.choice(np.flatnonzero(m)))
    local = play_page(LocalGames(PickTarget, 5), jobs, chooser())
    with ProcGames(PickTarget, 5, 2) as proc:
        assert play_page(proc, jobs, chooser()) == local


TESTS = [test_local_reset_shapes, test_local_reset_seeds, test_local_step_types, test_local_finished_games_restart,
    test_local_truncated_counts_as_done, test_local_masks_match_new_obs, test_proc_matches_local,
    test_proc_more_workers_than_games, test_proc_worker_error_is_raised, test_proc_close_stops_workers,
    test_jobs_free_slots_start_their_job, test_jobs_finished_game_takes_offer_or_goes_free,
    test_jobs_busy_slot_keeps_its_game, test_queue_offers_free_slots_first, test_queue_removes_only_started_jobs,
    test_page_plays_every_job_exactly_once, test_page_copies_with_same_moves_score_the_same,
    test_proc_page_matches_local]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")