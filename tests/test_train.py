import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from layout.gym_env import LayoutGym, StageEnv, N_POS
from layout.policy import Policy
from layout.testset import STAGES
from layout.runlog import read_log
from layout.train import train


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
    log = read_log("runs/_test/log.csv")
    assert np.array_equal(log["round"], [0, 1])
    for k in ["score", "failed", "invalid", "valid_score", "p_belt", "p_floors", "entropy", "approx_kl", "collect_s",
              "update_s", "gpu_gb"]:
        assert k in log, k
    assert np.allclose(log["score"], history, atol=1e-4, equal_nan=True)
    assert any(Path("runs/_test").glob("events.out.tfevents*"))
    shutil.rmtree("runs/_test")


def test_train_uses_cuda_when_available():
    if not torch.cuda.is_available():
        return
    shutil.rmtree("runs/_test_cuda", ignore_errors=True)
    policy, history = train(stage=1, rounds=1, n_games=2, workers=2, T=4, minibatch=8, epochs=1, run="_test_cuda",
                            verbose=False)
    assert next(policy.parameters()).device.type == "cuda"
    assert len(history) == 1 and read_log("runs/_test_cuda/log.csv")["gpu_gb"][0] > 0
    shutil.rmtree("runs/_test_cuda")


TESTS = [test_stage_env_builds_layout_gym, test_stage_env_only_stage_targets,
    test_stage_env_picklable, test_train_short_run_on_cpu, test_train_uses_cuda_when_available]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")