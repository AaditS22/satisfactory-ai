import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from layout.gym_env import LayoutGym, StageEnv, N_POS
from layout.policy import Policy
from layout.testset import STAGES
from layout.runlog import read_log
from layout.train import train, choose_greedy


def test_stage_env_builds_layout_gym():
    env = StageEnv(1)()
    assert isinstance(env, LayoutGym)
    obs, info = env.reset(seed=0)
    assert env.action_masks().shape == (N_POS,)


def test_stage_env_only_stage_targets():
    for stage in (1, 2):
        env = StageEnv(stage)()
        seen = set()
        for s in range(20):
            env.reset(seed=s)
            seen.add(env.task["target"])
        assert seen <= set(STAGES[stage - 1]) and len(seen) > 1


def test_stage_env_picklable():
    back = pickle.loads(pickle.dumps(StageEnv(2)))
    assert back.stage == 2 and back.targets == STAGES[1]


def test_choose_greedy_legal_and_repeatable():
    torch.manual_seed(0)
    policy = Policy()
    envs = [StageEnv(1)() for _ in range(3)]
    obs = [e.reset(seed=i)[0] for i, e in enumerate(envs)]
    masks = np.stack([e.action_masks() for e in envs])
    a1 = choose_greedy(policy, obs, masks)
    a2 = choose_greedy(policy, obs, masks)
    assert a1.shape == (3,) and np.array_equal(a1, a2) and masks[np.arange(3), a1].all()


def test_train_short_run_on_cpu():
    torch.manual_seed(0)
    fresh = [p.detach().clone() for p in Policy().parameters()]
    shutil.rmtree("runs/_test", ignore_errors=True)
    policy, history = train(stage=1, rounds=2, n_games=2, workers=2, T=6, minibatch=6, epochs=1, device="cpu",
                            run="_test", verbose=False)
    assert isinstance(policy, Policy)
    assert next(policy.parameters()).device.type == "cpu"
    assert isinstance(history, list) and len(history) == 2 and all(isinstance(h, float) for h in history)
    assert all(np.isnan(h) or -3.0 <= h <= 1.0 for h in history)
    assert any(not torch.equal(p.detach(), f) for p, f in zip(policy.parameters(), fresh))
    ev = read_log("runs/_test/eval.csv")
    assert np.array_equal(ev["round"], [0, 1])
    for k in ["score", "A", "B", "solv_score", "solv_A", "gate_A", "failed", "invalid", "eval_s"]:
        assert k in ev, k
    assert abs(ev["A"][0] - 0.653) < 0.001 and abs(ev["solv_A"][0] - 0.726) < 0.001
    assert len(list(Path("runs/_test/eval/round_0001").glob("best_*.png"))) == 10
    log = read_log("runs/_test/log.csv")
    assert np.array_equal(log["round"], [0, 1])
    for k in ["score", "failed", "invalid", "valid_score", "p_belt", "p_floors", "entropy", "approx_kl", "collect_s",
              "play_s", "update_s", "gpu_gb", "lr"]:
        assert k in log, k
    assert np.allclose(log["lr"], [2.5e-4, 1.25e-4])
    assert np.allclose(log["score"], history, atol=1e-4, equal_nan=True)
    assert len(list(Path("runs/_test").glob("events.out.tfevents*"))) == 1
    assert Path("runs/_test/last.pt").exists() and Path("runs/_test/best.pt").exists()
    last = torch.load("runs/_test/last.pt")
    assert last["round"] == 1 and last["extra"]["stage"] == 1 and last["extra"]["rounds"] == 2
    assert abs(last["extra"]["best_eval"] - np.max(ev["solv_score"])) < 1e-4
    shutil.rmtree("runs/_test")


def test_train_resume_continues():
    shutil.rmtree("runs/_test_resume", ignore_errors=True)
    p1, _ = train(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                  run="_test_resume", verbose=False)
    ck = torch.load("runs/_test_resume/last.pt")
    ck["extra"]["rounds"] = 3
    torch.save(ck, "runs/_test_resume/last.pt")
    p2, h2 = train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                   run="_test_resume", resume=True, verbose=False)
    assert len(h2) == 1
    assert np.array_equal(read_log("runs/_test_resume/log.csv")["round"], [0, 1, 2])
    assert np.array_equal(read_log("runs/_test_resume/eval.csv")["round"], [0, 1, 2])
    assert torch.load("runs/_test_resume/last.pt")["round"] == 2
    assert abs(read_log("runs/_test_resume/log.csv")["lr"][2] - 2.5e-4 / 3) < 1e-9
    shutil.rmtree("runs/_test_resume")


def test_resume_needs_run_name():
    try:
        train(rounds=1, resume=True, verbose=False)
        assert False
    except ValueError:
        pass


def test_resume_rejects_other_rounds():
    shutil.rmtree("runs/_test_rounds", ignore_errors=True)
    train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_rounds",
          verbose=False)
    try:
        train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_rounds",
              resume=True, verbose=False)
        assert False
    except ValueError:
        pass
    shutil.rmtree("runs/_test_rounds")


def test_init_and_resume_together_rejected():
    try:
        train(rounds=1, run="_test_both", resume=True, init="runs/_test_both/last.pt", verbose=False)
        assert False
    except ValueError:
        pass


def test_train_init_starts_new_run():
    shutil.rmtree("runs/_test_init_a", ignore_errors=True)
    shutil.rmtree("runs/_test_init_b", ignore_errors=True)
    train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_init_a",
          verbose=False)
    saved = torch.load("runs/_test_init_a/last.pt")["policy"]
    policy, history = train(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, lr=0.0, device="cpu",
                            run="_test_init_b", init="runs/_test_init_a/last.pt", verbose=False)
    assert len(history) == 2
    log = read_log("runs/_test_init_b/log.csv")
    assert np.array_equal(log["round"], [0, 1]) and np.array_equal(log["lr"], [0.0, 0.0])
    for k, v in policy.state_dict().items():
        assert torch.equal(v, saved[k]), k
    last = torch.load("runs/_test_init_b/last.pt")
    assert last["round"] == 1 and last["extra"]["rounds"] == 2
    shutil.rmtree("runs/_test_init_a")
    shutil.rmtree("runs/_test_init_b")


def test_train_uses_cuda_when_available():
    if not torch.cuda.is_available():
        return
    shutil.rmtree("runs/_test_cuda", ignore_errors=True)
    policy, history = train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, run="_test_cuda",
                            verbose=False)
    assert next(policy.parameters()).device.type == "cuda"
    assert len(history) == 1 and read_log("runs/_test_cuda/log.csv")["gpu_gb"][0] > 0
    shutil.rmtree("runs/_test_cuda")


def test_train_passes_solvable():
    import layout.train as LT
    real, seen = LT.StageEnv, []

    def spy(stage, solvable=False):
        seen.append((stage, solvable))
        return real(stage, solvable)

    shutil.rmtree("runs/_test_solvable", ignore_errors=True)
    LT.StageEnv = spy
    try:
        train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_solvable",
              solvable=True, verbose=False)
    finally:
        LT.StageEnv = real
    assert seen == [(1, True)]
    extra = torch.load("runs/_test_solvable/last.pt")["extra"]
    assert extra["solvable"] is True and extra["ent_coef"] == 0.01 and extra["stage"] == 1 and extra["rounds"] == 1
    shutil.rmtree("runs/_test_solvable")


def test_resume_rejects_other_settings():
    shutil.rmtree("runs/_test_settings", ignore_errors=True)
    base = dict(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_settings",
                verbose=False)
    train(**{**base, "rounds": 1}, ent_coef=0.0)
    ck = torch.load("runs/_test_settings/last.pt")
    ck["extra"]["rounds"] = 2
    torch.save(ck, "runs/_test_settings/last.pt")
    for bad in [{"stage": 2, "ent_coef": 0.0}, {"ent_coef": 0.01}, {"ent_coef": 0.0, "solvable": True},
                {"ent_coef": 0.0, "overlap": True}]:
        try:
            train(**{**base, **bad}, resume=True)
            assert False, bad
        except ValueError:
            pass
    _, h = train(**base, ent_coef=0.0, resume=True)
    assert len(h) == 1
    shutil.rmtree("runs/_test_settings")


def test_train_overlap_runs():
    torch.manual_seed(0)
    fresh = [p.detach().clone() for p in Policy().parameters()]
    shutil.rmtree("runs/_test_overlap", ignore_errors=True)
    policy, history = train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                            run="_test_overlap", overlap=True, verbose=False)
    assert len(history) == 3
    assert any(not torch.equal(p.detach(), f) for p, f in zip(policy.parameters(), fresh))
    log = read_log("runs/_test_overlap/log.csv")
    assert np.array_equal(log["round"], [0, 1, 2]) and (log["play_s"] > 0).all()
    extra = torch.load("runs/_test_overlap/last.pt")["extra"]
    assert extra["overlap"] is True and extra["rounds"] == 3
    shutil.rmtree("runs/_test_overlap")


TESTS = [test_stage_env_builds_layout_gym, test_stage_env_only_stage_targets, test_stage_env_picklable,
    test_choose_greedy_legal_and_repeatable, test_train_short_run_on_cpu, test_train_resume_continues,
    test_resume_needs_run_name, test_resume_rejects_other_rounds, test_init_and_resume_together_rejected,
    test_train_init_starts_new_run, test_train_uses_cuda_when_available, test_train_passes_solvable,
    test_resume_rejects_other_settings, test_train_overlap_runs]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")