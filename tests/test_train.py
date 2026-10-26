import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from layout.gym_env import LayoutGym, StageEnv, N_POS
from layout.policy import Policy
from layout.testset import STAGES
from layout.runlog import read_log
from layout.train import train, choose_greedy, page_jobs, is_broken, first_scale
from rl.buffer import EpisodeBuffer


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
    shutil.rmtree("runs/stage1/_test", ignore_errors=True)
    policy, history = train(stage=1, rounds=2, n_games=2, workers=2, T=6, minibatch=6, epochs=1, device="cpu",
                            run="_test", verbose=False)
    assert isinstance(policy, Policy)
    assert next(policy.parameters()).device.type == "cpu"
    assert isinstance(history, list) and len(history) == 2 and all(isinstance(h, float) for h in history)
    assert all(np.isnan(h) or -3.0 <= h <= 1.0 for h in history)
    assert any(not torch.equal(p.detach(), f) for p, f in zip(policy.parameters(), fresh))
    ev = read_log("runs/stage1/_test/eval.csv")
    assert np.array_equal(ev["round"], [0, 1])
    for k in ["score", "A", "B", "solv_score", "solv_A", "gate_A", "failed", "invalid", "eval_s"]:
        assert k in ev, k
    assert abs(ev["A"][0] - 0.653) < 0.001 and abs(ev["solv_A"][0] - 0.726) < 0.001
    assert len(list(Path("runs/stage1/_test/eval/round_0001").glob("best_*.png"))) == 10
    log = read_log("runs/stage1/_test/log.csv")
    assert np.array_equal(log["round"], [0, 1])
    for k in ["score", "failed", "invalid", "valid_score", "p_belt", "p_floors", "entropy", "approx_kl", "collect_s",
              "play_s", "update_s", "gpu_gb", "lr", "shaping", "wire"]:
        assert k in log, k
    assert np.allclose(log["lr"], [2.5e-4, 1.25e-4])
    assert np.allclose(log["score"], history, atol=1e-4, equal_nan=True)
    assert len(list(Path("runs/stage1/_test").glob("events.out.tfevents*"))) == 1
    assert Path("runs/stage1/_test/last.pt").exists() and Path("runs/stage1/_test/best.pt").exists()
    last = torch.load("runs/stage1/_test/last.pt")
    assert last["round"] == 1 and last["extra"]["stage"] == 1 and last["extra"]["rounds"] == 2
    assert abs(last["extra"]["best_eval"] - np.max(ev["solv_score"])) < 1e-4
    shutil.rmtree("runs/stage1/_test")


def test_train_resume_continues():
    shutil.rmtree("runs/stage1/_test_resume", ignore_errors=True)
    p1, _ = train(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                  run="_test_resume", verbose=False)
    ck = torch.load("runs/stage1/_test_resume/last.pt")
    ck["extra"]["rounds"] = 3
    torch.save(ck, "runs/stage1/_test_resume/last.pt")
    p2, h2 = train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                   run="_test_resume", resume=True, verbose=False)
    assert len(h2) == 1
    assert np.array_equal(read_log("runs/stage1/_test_resume/log.csv")["round"], [0, 1, 2])
    assert np.array_equal(read_log("runs/stage1/_test_resume/eval.csv")["round"], [0, 1, 2])
    assert torch.load("runs/stage1/_test_resume/last.pt")["round"] == 2
    assert abs(read_log("runs/stage1/_test_resume/log.csv")["lr"][2] - 2.5e-4 / 3) < 1e-9
    shutil.rmtree("runs/stage1/_test_resume")


def test_resume_needs_run_name():
    try:
        train(rounds=1, resume=True, verbose=False)
        assert False
    except ValueError:
        pass


def test_resume_rejects_other_rounds():
    shutil.rmtree("runs/stage1/_test_rounds", ignore_errors=True)
    train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_rounds",
          verbose=False)
    try:
        train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_rounds",
              resume=True, verbose=False)
        assert False
    except ValueError:
        pass
    shutil.rmtree("runs/stage1/_test_rounds")


def test_init_and_resume_together_rejected():
    try:
        train(rounds=1, run="_test_both", resume=True, init="runs/stage1/_test_both/last.pt", verbose=False)
        assert False
    except ValueError:
        pass


def test_train_init_starts_new_run():
    shutil.rmtree("runs/stage1/_test_init_a", ignore_errors=True)
    shutil.rmtree("runs/stage1/_test_init_b", ignore_errors=True)
    train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_init_a",
          verbose=False)
    saved = torch.load("runs/stage1/_test_init_a/last.pt")["policy"]
    policy, history = train(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, lr=0.0, device="cpu",
                            run="_test_init_b", init="runs/stage1/_test_init_a/last.pt", verbose=False)
    assert len(history) == 2
    log = read_log("runs/stage1/_test_init_b/log.csv")
    assert np.array_equal(log["round"], [0, 1]) and np.array_equal(log["lr"], [0.0, 0.0])
    for k, v in policy.state_dict().items():
        assert torch.equal(v, saved[k]), k
    last = torch.load("runs/stage1/_test_init_b/last.pt")
    assert last["round"] == 1 and last["extra"]["rounds"] == 2
    shutil.rmtree("runs/stage1/_test_init_a")
    shutil.rmtree("runs/stage1/_test_init_b")


def test_train_uses_cuda_when_available():
    if not torch.cuda.is_available():
        return
    shutil.rmtree("runs/stage1/_test_cuda", ignore_errors=True)
    policy, history = train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, run="_test_cuda",
                            verbose=False)
    assert next(policy.parameters()).device.type == "cuda"
    assert len(history) == 1 and read_log("runs/stage1/_test_cuda/log.csv")["gpu_gb"][0] > 0
    shutil.rmtree("runs/stage1/_test_cuda")


def test_train_passes_solvable():
    import layout.train as LT
    real, seen = LT.StageEnv, []

    def spy(stage, solvable=False):
        seen.append((stage, solvable))
        return real(stage, solvable)

    shutil.rmtree("runs/stage1/_test_solvable", ignore_errors=True)
    LT.StageEnv = spy
    try:
        train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_solvable",
              solvable=True, verbose=False)
    finally:
        LT.StageEnv = real
    assert seen == [(1, True)]
    extra = torch.load("runs/stage1/_test_solvable/last.pt")["extra"]
    assert extra["solvable"] is True and extra["ent_coef"] == 0.01 and extra["stage"] == 1 and extra["rounds"] == 1
    shutil.rmtree("runs/stage1/_test_solvable")


def test_resume_rejects_other_settings():
    shutil.rmtree("runs/stage1/_test_settings", ignore_errors=True)
    base = dict(stage=1, rounds=2, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_settings",
                verbose=False)
    train(**{**base, "rounds": 1}, ent_coef=0.0)
    ck = torch.load("runs/stage1/_test_settings/last.pt")
    ck["extra"]["rounds"] = 2
    torch.save(ck, "runs/stage1/_test_settings/last.pt")
    for bad in [{"stage": 2, "ent_coef": 0.0}, {"ent_coef": 0.01}, {"ent_coef": 0.0, "solvable": True},
                {"ent_coef": 0.0, "overlap": True}, {"ent_coef": 0.0, "shaping": 0.1}, {"ent_coef": 0.0, "copies": 2},
                {"ent_coef": 0.0, "groups": 10}, {"ent_coef": 0.0, "broken_margin": 0.2}]:
        try:
            train(**{**base, **bad}, resume=True)
            assert False, bad
        except ValueError:
            pass
    _, h = train(**base, ent_coef=0.0, resume=True)
    assert len(h) == 1
    shutil.rmtree("runs/stage1/_test_settings")


def test_train_overlap_runs():
    torch.manual_seed(0)
    fresh = [p.detach().clone() for p in Policy().parameters()]
    shutil.rmtree("runs/stage1/_test_overlap", ignore_errors=True)
    policy, history = train(stage=1, rounds=3, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu",
                            run="_test_overlap", overlap=True, verbose=False)
    assert len(history) == 3
    assert any(not torch.equal(p.detach(), f) for p, f in zip(policy.parameters(), fresh))
    log = read_log("runs/stage1/_test_overlap/log.csv")
    assert np.array_equal(log["round"], [0, 1, 2]) and (log["play_s"] > 0).all()
    extra = torch.load("runs/stage1/_test_overlap/last.pt")["extra"]
    assert extra["overlap"] is True and extra["rounds"] == 3
    shutil.rmtree("runs/stage1/_test_overlap")


def test_train_shaping_rewards_and_log():
    import layout.train as LT
    real, pages = LT.update, []

    def spy(policy, optimizer, buffer, *a, **k):
        pages.append((buffer.rewards.copy(), buffer.dones.copy()))
        return real(policy, optimizer, buffer, *a, **k)

    shutil.rmtree("runs/stage1/_test_shaping", ignore_errors=True)
    LT.update = spy
    try:
        train(stage=1, rounds=2, n_games=2, workers=2, T=8, minibatch=8, epochs=1, device="cpu", run="_test_shaping",
              shaping=0.1, verbose=False)
    finally:
        LT.update = real
    (r0, d0), (r1, d1) = pages
    assert (r0[~d0] <= 0).all() and (r0[~d0] < 0).any()
    assert (r1[~d1] == 0).all()
    log = read_log("runs/stage1/_test_shaping/log.csv")
    assert np.allclose(log["shaping"], [0.1, 0.0]) and "wire" in log
    assert torch.load("runs/stage1/_test_shaping/last.pt")["extra"]["shaping"] == 0.1
    shutil.rmtree("runs/stage1/_test_shaping")


def test_train_images_every_zero_means_never():
    shutil.rmtree("runs/stage1/_test_noimg", ignore_errors=True)
    train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_noimg",
          images_every=0, verbose=False)
    assert not list(Path("runs/stage1/_test_noimg").glob("eval/**/*.png"))
    assert Path("runs/stage1/_test_noimg/last.pt").exists()
    shutil.rmtree("runs/stage1/_test_noimg")


def test_run_json_and_existing_run_refused():
    shutil.rmtree("runs/stage1/_test_note", ignore_errors=True)
    base = dict(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, device="cpu", run="_test_note",
                verbose=False)
    train(**base, shaping=0.1, note="try shaping")
    info = json.loads(Path("runs/stage1/_test_note/run.json").read_text())
    assert info["note"] == "try shaping" and info["config"]["shaping"] == 0.1 and info["config"]["rounds"] == 1
    assert info["config"]["init"] is None and info["resumed"] == []
    try:
        train(**base, shaping=0.1, note="again")
        assert False
    except ValueError:
        pass
    ck = torch.load("runs/stage1/_test_note/last.pt")
    ck["extra"]["rounds"] = 2
    torch.save(ck, "runs/stage1/_test_note/last.pt")
    train(**{**base, "rounds": 2}, shaping=0.1, resume=True, note="after a crash")
    info = json.loads(Path("runs/stage1/_test_note/run.json").read_text())
    assert info["note"] == "try shaping" and [r["at_round"] for r in info["resumed"]] == [1]
    shutil.rmtree("runs/stage1/_test_note")


def test_page_jobs():
    assert page_jobs(0, 0, 3, 2) == [1_000_000, 1_000_000, 1_000_001, 1_000_001, 1_000_002, 1_000_002]
    assert page_jobs(0, 1, 3, 2)[0] == 1_000_003
    jobs = page_jobs(1, 5, 40, 8)
    assert len(jobs) == 320 and jobs[0] == 2_000_200 and jobs[7] == 2_000_200 and jobs[8] == 2_000_201
    seen = {j for r in range(2000) for j in page_jobs(0, r, 40, 1)}
    assert len(seen) == 80_000 and min(seen) == 1_000_000 and max(seen) < 2_000_000


def test_is_broken():
    assert is_broken({"failed": True})
    assert is_broken({"failed": False, "terms": {"passed": False}})
    assert not is_broken({"failed": False, "terms": {"passed": True}})


def test_train_groups_short_run():
    import layout.train as LT
    real_c, real_u, jobs_seen, pages = LT.collect_groups, LT.update, [], []

    def spy_c(games, policy, buffer, jobs, *a, **k):
        jobs_seen.append(list(jobs))
        return real_c(games, policy, buffer, jobs, *a, **k)

    def spy_u(policy, optimizer, buffer, *a, **k):
        _, d = buffer.get(np.arange(buffer.T))
        pages.append((type(buffer), buffer.T, buffer.N, d["adv"].copy(), d["ret"].copy()))
        return real_u(policy, optimizer, buffer, *a, **k)

    shutil.rmtree("runs/stage1/_test_groups", ignore_errors=True)
    LT.collect_groups, LT.update = spy_c, spy_u
    try:
        torch.manual_seed(0)
        _, history = train(stage=1, rounds=2, n_games=4, workers=2, minibatch=8, epochs=1, device="cpu",
                           run="_test_groups", copies=2, groups=3, verbose=False)
    finally:
        LT.collect_groups, LT.update = real_c, real_u
    assert jobs_seen == [page_jobs(0, 0, 3, 2), page_jobs(0, 1, 3, 2)]
    for kind_, T, N, adv, ret in pages:
        assert kind_ is EpisodeBuffer and N == 1 and T == len(adv) >= 12 and np.isfinite(adv).all()
    log = read_log("runs/stage1/_test_groups/log.csv")
    assert np.array_equal(log["games"], [6, 6]) and np.array_equal(log["groups"], [3, 3])
    assert np.array_equal(log["steps"], [p[1] for p in pages])
    assert ((log["idle"] >= 0) & (log["idle"] < 1)).all() and ((log["flat"] >= 0) & (log["flat"] <= 1)).all()
    assert "adv" in log and len(history) == 2
    extra = torch.load("runs/stage1/_test_groups/last.pt")["extra"]
    assert extra["copies"] == 2 and extra["groups"] == 3 and extra["broken_margin"] == 0.1
    shutil.rmtree("runs/stage1/_test_groups")


def test_groups_refuse_overlap_and_shaping():
    for bad in [{"overlap": True}, {"shaping": 0.1}]:
        try:
            train(stage=1, rounds=1, n_games=2, workers=2, device="cpu", run="_test_refuse", copies=2, verbose=False,
                  **bad)
            assert False, bad
        except ValueError:
            pass
    assert not Path("runs/stage1/_test_refuse").exists()


def test_first_scale():
    assert abs(first_scale(np.array([0.01, -0.01, 0.01, -0.01], np.float32)) - 0.0115470) < 1e-6
    assert first_scale(np.zeros(5, np.float32)) is None
    assert first_scale(np.array([0.3], np.float32)) is None


def test_fixed_scale_needs_copies():
    try:
        train(stage=1, rounds=1, n_games=2, workers=2, device="cpu", run="_test_refuse", fixed_scale=True, verbose=False)
        assert False
    except ValueError:
        pass
    assert not Path("runs/stage1/_test_refuse").exists()


def test_train_fixed_scale_frozen_and_resumed():
    import layout.train as LT
    real_u, seen = LT.update, []

    def spy_u(policy, optimizer, buffer, *a, **k):
        seen.append((k.get("adv_scale"), buffer.adv.copy()))
        return real_u(policy, optimizer, buffer, *a, **k)

    shutil.rmtree("runs/stage1/_test_scale", ignore_errors=True)
    base = dict(stage=1, rounds=2, n_games=4, workers=2, minibatch=8, epochs=1, device="cpu", run="_test_scale",
                copies=2, groups=3, fixed_scale=True, verbose=False)
    LT.update = spy_u
    try:
        torch.manual_seed(0)
        train(**base)
        ck = torch.load("runs/stage1/_test_scale/last.pt")
        ck["extra"]["rounds"] = 3
        torch.save(ck, "runs/stage1/_test_scale/last.pt")
        try:
            train(**{**base, "rounds": 3, "fixed_scale": False}, resume=True)
            assert False
        except ValueError:
            pass
        train(**{**base, "rounds": 3}, resume=True)
    finally:
        LT.update = real_u
    want = float(np.std(seen[0][1], ddof=1))
    assert want > 0 and abs(first_scale(seen[0][1]) - want) < 1e-12
    assert len(seen) == 3 and all(s == want for s, _ in seen)
    assert not np.allclose(np.std(seen[2][1], ddof=1), want)
    log = read_log("runs/stage1/_test_scale/log.csv")
    assert np.array_equal(log["round"], [0, 1, 2]) and np.allclose(log["scale"], want, rtol=1e-4)
    assert (log["push"] > 0).all()
    extra = torch.load("runs/stage1/_test_scale/last.pt")["extra"]
    assert extra["fixed_scale"] is True and abs(extra["adv_scale"] - want) < 1e-12
    shutil.rmtree("runs/stage1/_test_scale")


def test_train_without_fixed_scale_uses_curve():
    import layout.train as LT
    real_u, seen = LT.update, []

    def spy_u(policy, optimizer, buffer, *a, **k):
        seen.append(k.get("adv_scale", "missing"))
        return real_u(policy, optimizer, buffer, *a, **k)

    shutil.rmtree("runs/stage1/_test_curve", ignore_errors=True)
    LT.update = spy_u
    try:
        train(stage=1, rounds=1, n_games=4, workers=2, minibatch=8, epochs=1, device="cpu", run="_test_curve",
              copies=2, groups=3, verbose=False)
    finally:
        LT.update = real_u
    assert seen == [None]
    assert "scale" not in read_log("runs/stage1/_test_curve/log.csv")
    shutil.rmtree("runs/stage1/_test_curve")


def test_scale_refusals():
    base = dict(stage=1, rounds=1, n_games=2, workers=2, device="cpu", run="_test_refuse", copies=2, verbose=False)
    for bad in [{"set_scale": 0.05}, {"fixed_scale": True, "set_scale": 0.0}, {"fixed_scale": True, "set_scale": -0.1},
                {"fixed_scale": True, "set_scale": 0.05, "resume": True}]:
        try:
            train(**base, **bad)
            assert False, bad
        except ValueError:
            pass
    assert not Path("runs/stage1/_test_refuse").exists()


def test_train_set_scale_used_from_round_0():
    import layout.train as LT
    real_u, seen = LT.update, []

    def spy_u(policy, optimizer, buffer, *a, **k):
        seen.append(k.get("adv_scale"))
        return real_u(policy, optimizer, buffer, *a, **k)

    shutil.rmtree("runs/stage1/_test_setscale", ignore_errors=True)
    LT.update = spy_u
    try:
        train(stage=1, rounds=2, n_games=4, workers=2, minibatch=8, epochs=1, device="cpu", run="_test_setscale",
              copies=2, groups=3, fixed_scale=True, set_scale=0.05, verbose=False)
    finally:
        LT.update = real_u
    assert seen == [0.05, 0.05]
    log = read_log("runs/stage1/_test_setscale/log.csv")
    assert np.allclose(log["scale"], 0.05)
    extra = torch.load("runs/stage1/_test_setscale/last.pt")["extra"]
    assert extra["adv_scale"] == 0.05 and "set_scale" not in extra
    info = json.loads(Path("runs/stage1/_test_setscale/run.json").read_text())
    assert info["config"]["set_scale"] == 0.05
    shutil.rmtree("runs/stage1/_test_setscale")


def run_main(argv):
    import layout.train as LT
    real, seen = LT.train, {}

    def spy(**kw):
        seen.update(kw)
        return None, [0.5]

    old = sys.argv
    LT.train, sys.argv = spy, ["layout.train"] + argv
    try:
        LT.main()
    finally:
        LT.train, sys.argv = real, old
    return seen


def test_main_passes_every_flag():
    seen = run_main(["--run", "x", "--stage", "2", "--rounds", "7", "--ent-coef", "0.03", "--workers", "3",
                     "--solvable", "--overlap", "--shaping", "0.1", "--lr", "0.001", "--images-every", "0",
                     "--init", "a.pt", "--note", "hello", "--copies", "8", "--groups", "20", "--broken-margin", "0.2",
                     "--fixed-scale", "--scale", "0.05"])
    want = {"run": "x", "stage": 2, "rounds": 7, "ent_coef": 0.03, "workers": 3, "solvable": True, "overlap": True,
            "shaping": 0.1, "lr": 0.001, "images_every": 0, "init": "a.pt", "note": "hello", "resume": False,
            "copies": 8, "groups": 20, "broken_margin": 0.2, "fixed_scale": True, "set_scale": 0.05}
    for k, v in want.items():
        assert seen[k] == v, k


def test_main_needs_note_for_new_runs():
    try:
        run_main(["--run", "x"])
        assert False
    except SystemExit:
        pass
    assert run_main(["--run", "x", "--resume"])["resume"] is True


TESTS = [test_stage_env_builds_layout_gym, test_stage_env_only_stage_targets, test_stage_env_picklable,
    test_choose_greedy_legal_and_repeatable, test_train_short_run_on_cpu, test_train_resume_continues,
    test_resume_needs_run_name, test_resume_rejects_other_rounds, test_init_and_resume_together_rejected,
    test_train_init_starts_new_run, test_train_uses_cuda_when_available, test_train_passes_solvable,
    test_resume_rejects_other_settings, test_train_overlap_runs, test_train_shaping_rewards_and_log,
    test_train_images_every_zero_means_never, test_run_json_and_existing_run_refused, test_main_passes_every_flag,
    test_main_needs_note_for_new_runs, test_page_jobs, test_is_broken, test_train_groups_short_run,
    test_groups_refuse_overlap_and_shaping, test_first_scale, test_fixed_scale_needs_copies,
    test_train_fixed_scale_frozen_and_resumed, test_train_without_fixed_scale_uses_curve, test_scale_refusals,
    test_train_set_scale_used_from_round_0]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")