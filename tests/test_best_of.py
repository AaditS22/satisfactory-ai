import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from layout.best_of import keep_best, compare, play_jobs, as_results, save_plans, TaskGym, EvalTasks
from layout.evaluate import load_eval, play_eval
from layout.api import evaluate
from rl.games import LocalGames


def res(score, A=0.5, B=0.6, kind="valid"):
    return {"score": score, "A": A, "B": B, "kind": kind}


class Count:
    def __init__(self):
        self.left, self.job, self.moves = 0, None, []

    def reset(self, seed=None, options=None):
        self.job, self.left, self.moves = seed, seed % 3 + 1, []
        return {"x": np.zeros(1)}, {}

    def step(self, a):
        self.moves.append(int(a))
        self.left -= 1
        done = self.left == 0
        return {"x": np.zeros(1)}, float(self.job * 10 + len(self.moves)) if done else 0.0, done, False, {"job": self.job}

    def action_masks(self):
        return np.ones(2, dtype=bool)


def first_legal(obs_list, masks):
    return [int(np.flatnonzero(m)[0]) for m in masks]


def rows_and_tasks(stage=1, n=4):
    rows = load_eval(stage)[:n]
    return rows, [r["task"] for r in rows]


def test_keep_best_per_task():
    tries = [[res(0.1), res(0.7), res(-1.2, kind="failed")],
             [res(0.4), res(0.2), res(0.3)],
             [res(0.3), res(0.6), res(-0.5, kind="invalid")]]
    best = keep_best(tries)
    assert [b["score"] for b in best] == [0.4, 0.7, 0.3]
    assert best[2]["kind"] == "valid"


def test_keep_best_one_try_unchanged():
    one = [res(0.1), res(0.2)]
    assert keep_best([one]) == one


def test_keep_best_does_not_change_input():
    tries = [[res(0.1)], [res(0.9)]]
    keep_best(tries)
    assert tries[0][0]["score"] == 0.1


def test_compare_hand():
    results = [res(0.6, A=0.5, B=0.6), res(0.5, A=0.5, B=0.7), res(0.4, A=0.5, B=0.55),
               res(-1.2, A=-1.25, B=-1.25, kind="failed")]
    s = compare(results)
    assert s["solvable"] == 3
    assert abs(s["beats_A"] - 1 / 3) < 1e-9 and abs(s["ties_A"] - 1 / 3) < 1e-9
    assert abs(s["beats_B"] - 1 / 3) < 1e-9 and s["invalid"] == 0
    assert abs(s["score"] - (0.6 + 0.5 + 0.4) / 3) < 1e-9


def test_compare_counts_invalid_on_solvable_only():
    s = compare([res(-0.4, kind="invalid"), res(0.7), res(-1.3, A=-1.4, kind="failed")])
    assert s["solvable"] == 2 and s["invalid"] == 0.5


def test_play_jobs_every_job_once():
    for slots in [1, 3, 5, 20]:
        jobs = list(range(100, 112))
        out = play_jobs(LocalGames(Count, slots), jobs, first_legal)
        assert sorted(out) == jobs, slots
        for j, info in out.items():
            assert info["score"] == j * 10 + j % 3 + 1 and info["job"] == j, (slots, j)


def test_play_jobs_empty():
    assert play_jobs(LocalGames(Count, 3), [], first_legal) == {}


def test_play_jobs_choose_gets_only_live_games():
    seen = []

    def spy(obs_list, masks):
        seen.append((len(obs_list), masks.shape[0]))
        return first_legal(obs_list, masks)

    play_jobs(LocalGames(Count, 4), [0, 1, 2, 3, 4, 5], spy)
    assert seen and all(a == b and 1 <= a <= 4 for a, b in seen)


def test_task_gym_reset_picks_task_by_job():
    rows, tasks = rows_and_tasks()
    env = TaskGym(tasks)
    for job in [0, 2, 5, 11]:
        env.reset(seed=job)
        assert env.task is tasks[job % 4]


def test_task_gym_plan_replays_to_same_score():
    rows, tasks = rows_and_tasks()
    env = TaskGym(tasks)
    for job in range(4):
        env.reset(seed=job)
        done = False
        while not done:
            _, reward, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert "plan" in info and len(info["plan"]) == info["blocks"]
        assert abs(evaluate(tasks[job], info["plan"])["score"] - reward) < 1e-12


def test_eval_tasks_picklable():
    _, tasks = rows_and_tasks()
    maker = pickle.loads(pickle.dumps(EvalTasks(tasks)))
    env = maker()
    env.reset(seed=1)
    assert env.task["target"] == tasks[1]["target"] and env.task["rate"] == tasks[1]["rate"]


def test_slots_match_play_eval():
    rows, tasks = rows_and_tasks(n=6)
    want = play_eval(None, rows, lambda policy, obs_list, masks: first_legal(obs_list, masks))
    for slots in [2, 4]:
        infos = play_jobs(LocalGames(EvalTasks(tasks), slots), list(range(6, 18)), first_legal)
        for first in [6, 12]:
            got = as_results(rows, infos, first)
            assert [x["score"] for x in got] == [x["score"] for x in want], (slots, first)
            assert [x["kind"] for x in got] == [x["kind"] for x in want]
            assert [x["A"] for x in got] == [r["A"]["score"] for r in rows]


def test_save_plans_writes_plans():
    rows, tasks = rows_and_tasks(n=2)
    infos = play_jobs(LocalGames(EvalTasks(tasks), 2), [0, 1], first_legal)
    g = as_results(rows, infos, 0)
    b = [dict(x, plan=[[9, 9, 9, 9, 9]], score=0.99) for x in g]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.json"
        save_plans(path, rows, g, b)
        out = json.loads(path.read_text())
    assert [o["seed"] for o in out] == [r["seed"] for r in rows]
    for o, x, t in zip(out, g, tasks):
        assert o["greedy"]["plan"] == x["plan"] and o["greedy"]["score"] == x["score"]
        assert o["best"]["plan"] == [[9, 9, 9, 9, 9]] and o["best"]["score"] == 0.99
        assert abs(evaluate(t, o["greedy"]["plan"])["score"] - x["score"]) < 1e-12


TESTS = [test_keep_best_per_task, test_keep_best_one_try_unchanged, test_keep_best_does_not_change_input,
    test_compare_hand, test_compare_counts_invalid_on_solvable_only, test_play_jobs_every_job_once,
    test_play_jobs_empty, test_play_jobs_choose_gets_only_live_games, test_task_gym_reset_picks_task_by_job,
    test_task_gym_plan_replays_to_same_score, test_eval_tasks_picklable, test_slots_match_play_eval,
    test_save_plans_writes_plans]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")