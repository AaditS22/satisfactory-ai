import json
import tempfile
from pathlib import Path

import numpy as np

from layout.baseline import baseline_a
from layout.gym_env import LayoutGym, POS_SHAPE, S, N_POS
from layout.teacher import teacher_task, run_one, load_teacher, factory_key, steps, split_rows, build, batch, GRID_SHAPE, SMALL_KEYS


def plan_for(seed):
    task = teacher_task(1, seed)
    return task, baseline_a(task)["plan"]


def test_teacher_task_repeatable():
    a = teacher_task(1, 7)
    b = teacher_task(1, 7)
    c = teacher_task(1, 8)
    assert a == b
    assert a != c
    assert a["target"] in ("Desc_IronPlate_C", "Desc_IronRod_C", "Desc_Wire_C")


def test_steps_two_per_block_with_stage_flags():
    task, plan = plan_for(1)
    out = steps(task, plan)
    assert len(out) == 2 * len(plan)
    for k, (obs, mask, a) in enumerate(out):
        assert obs["stage"][0] == k % 2
        assert mask.shape == (int(np.prod(POS_SHAPE)),) and mask.dtype == bool


def test_steps_n_actions_are_plan_n():
    task, plan = plan_for(2)
    out = steps(task, plan)
    assert [out[2 * i][2] for i in range(len(plan))] == [p[0] for p in plan]
    for i, p in enumerate(plan):
        assert abs(out[2 * i + 1][0]["stage"][1] - p[0] / 16) < 1e-6


def test_steps_actions_always_legal():
    for seed in range(4):
        task, plan = plan_for(seed)
        for obs, mask, a in steps(task, plan):
            assert mask[a]


def test_steps_position_square_hand():
    task, plan = plan_for(3)
    n, rot, f, x, y = plan[0]
    a = steps(task, plan)[1][2]
    assert np.unravel_index(a, POS_SHAPE) == (rot, f, x // S, y // S)


def test_steps_replay_through_gym_lands_in_same_squares():
    task, plan = plan_for(5)
    gym = LayoutGym()
    gym.reset(options={"task": task})
    done = False
    for obs, mask, a in steps(task, plan):
        assert np.array_equal(gym.action_masks(), mask)
        _, r, done, _, info = gym.step(a)
    assert done
    placed = gym.env.world.blocks
    assert len(placed) == len(plan)
    for b, (n, rot, f, x, y) in zip(placed, plan):
        assert b["block"]["n"] == n and b["floor"] == f and b["block"]["rot"] == rot
        assert (b["x"] // S, b["y"] // S) == (x // S, y // S)


def test_steps_first_obs_matches_gym_obs():
    task, plan = plan_for(6)
    gym = LayoutGym()
    obs, _ = gym.reset(options={"task": task})
    first = steps(task, plan)[0][0]
    assert sorted(obs) == sorted(first)
    for k in obs:
        assert np.array_equal(obs[k], first[k])


def test_run_one_record_is_plain_json():
    r = run_one((1, 3, 5))
    text = json.dumps(r)
    back = json.loads(text)
    assert back["seed"] == 3 and isinstance(back["task"]["alternates"], list)
    assert back["score"] == r["score"] and len(back["plan"]) >= 1


def eval_like_row(seed):
    task = teacher_task(1, seed)
    t = dict(task)
    t["alternates"] = sorted(t["alternates"])
    return t


def test_load_teacher_keeps_working_layouts_only():
    rows = [{"seed": 0, "task": eval_like_row(0), "plan": [[1, 0, 0, 0, 0]], "score": 0.7},
            {"seed": 1, "task": eval_like_row(1), "plan": [], "score": -1.2},
            {"seed": 2, "task": eval_like_row(2), "plan": [], "score": -0.3}]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.json"
        path.write_text(json.dumps(rows))
        ev = Path(d) / "e.json"
        ev.write_text(json.dumps([{"task": eval_like_row(9)}]))
        kept = load_teacher(path, ev)
    assert [r["seed"] for r in kept] == [0]
    assert kept[0]["task"]["alternates"] == frozenset(eval_like_row(0)["alternates"])


def test_load_teacher_drops_eval_tasks():
    same = eval_like_row(4)
    renamed = dict(same)
    renamed["alternates"] = sorted(set(same["alternates"]) | {"Recipe_Alternate_NotUsedHere_C"})
    rows = [{"seed": 3, "task": eval_like_row(3), "plan": [[1, 0, 0, 0, 0]], "score": 0.7},
            {"seed": 4, "task": renamed, "plan": [[1, 0, 0, 0, 0]], "score": 0.7}]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.json"
        path.write_text(json.dumps(rows))
        ev = Path(d) / "e.json"
        ev.write_text(json.dumps([{"task": same}]))
        kept = load_teacher(path, ev)
    assert [r["seed"] for r in kept] == [3]


def test_factory_key_ignores_alternate_names_but_not_site():
    t = teacher_task(1, 5)
    a = dict(t, alternates=frozenset(t["alternates"]) | {"Recipe_Alternate_NotUsedHere_C"})
    b = dict(t, width=t["width"] + 8)
    assert factory_key(t) == factory_key(a)
    assert factory_key(t) != factory_key(b)


def record(seed):
    task = teacher_task(1, seed)
    a = baseline_a(task)
    return {"seed": seed, "task": task, "plan": a["plan"], "score": float(a["score"])}


ROWS = [record(s) for s in range(3)]


def test_split_rows_holds_out_last_seeds():
    rows = [{"seed": s} for s in [5, 1, 9, 3, 7, 0, 2, 8, 4, 6]]
    train, val = split_rows(rows, val_frac=0.2)
    assert [r["seed"] for r in val] == [8, 9]
    assert [r["seed"] for r in train] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_split_rows_keeps_at_least_one_val():
    train, val = split_rows([{"seed": 0}, {"seed": 1}, {"seed": 2}], val_frac=0.1)
    assert len(val) == 1 and len(train) == 2


def test_build_row_count_and_shapes():
    s = build(ROWS)
    n = sum(2 * len(r["plan"]) for r in ROWS)
    assert s["grid_bits"].shape == (n, int(np.prod(GRID_SHAPE)) // 8)
    assert s["mask_bits"].shape == (n, N_POS // 8)
    assert s["action"].shape == (n,) and s["action"].dtype == np.int64
    assert s["score"].shape == (n,) and s["score"].dtype == np.float32
    for k in SMALL_KEYS:
        assert len(s[k]) == n


def test_batch_unpacks_exactly():
    s = build(ROWS)
    want = []
    for r in ROWS:
        for o, m, a in steps(r["task"], r["plan"]):
            want.append((o, m, a, r["score"]))
    idx = np.array([len(want) - 1, 0, 2])
    obs, masks, actions, scores = batch(s, idx)
    assert obs["grid"].shape == (3, *GRID_SHAPE) and obs["grid"].dtype == np.uint8
    assert masks.shape == (3, N_POS) and masks.dtype == bool
    for j, i in enumerate(idx):
        o, m, a, sc = want[i]
        for k in o:
            assert np.array_equal(obs[k][j], o[k])
        assert np.array_equal(masks[j], m)
        assert actions[j] == a and abs(scores[j] - sc) < 1e-6


def test_batch_actions_are_legal():
    s = build(ROWS)
    obs, masks, actions, scores = batch(s, np.arange(len(s["action"])))
    assert masks[np.arange(len(actions)), actions].all()


def test_score_same_for_every_step_of_a_task():
    s = build(ROWS[:1])
    assert np.all(s["score"] == np.float32(ROWS[0]["score"]))


TESTS = [test_teacher_task_repeatable, test_steps_two_per_block_with_stage_flags, test_steps_n_actions_are_plan_n,
    test_steps_actions_always_legal, test_steps_position_square_hand, test_steps_replay_through_gym_lands_in_same_squares,
    test_steps_first_obs_matches_gym_obs, test_run_one_record_is_plain_json, test_load_teacher_keeps_working_layouts_only,
    test_load_teacher_drops_eval_tasks, test_factory_key_ignores_alternate_names_but_not_site,
    test_split_rows_holds_out_last_seeds, test_split_rows_keeps_at_least_one_val, test_build_row_count_and_shapes,
    test_batch_unpacks_exactly, test_batch_actions_are_legal, test_score_same_for_every_step_of_a_task]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")