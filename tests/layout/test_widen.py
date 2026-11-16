import tempfile
from pathlib import Path

import numpy as np
import torch

from layout.widen import widen, save_wide
from layout.policy import Policy, obs_to_torch
from layout.gym_env import LayoutGym
from layout.evaluate import load_eval
from rl.checkpoint import save_checkpoint, load_checkpoint
from rl.collect import stack
from rl.dist import masked_log_probs


def two_obs():
    gym = LayoutGym()
    o0, _ = gym.reset(options={"task": load_eval(1)[0]["task"]})
    m0 = gym.action_masks()
    o1, _, _, _, _ = gym.step(int(np.flatnonzero(m0)[-1]))
    m1 = gym.action_masks()
    return [o0, o1], np.stack([m0, m1])


def outputs_of(policy, obs_list):
    with torch.no_grad():
        return policy(obs_to_torch(stack(obs_list)))


def fresh(seed=0):
    torch.manual_seed(seed)
    return Policy().eval()


def test_rows_above_src_copy_src():
    p = fresh()
    widen(p, 8)
    w, b = p.summary.n_head.weight, p.summary.n_head.bias
    assert w.shape[0] == 17
    for n in range(9, 17):
        assert torch.equal(w[n], w[8])
        assert torch.equal(b[n], b[8])


def test_rows_up_to_src_unchanged():
    p = fresh()
    w0 = p.summary.n_head.weight.clone()
    b0 = p.summary.n_head.bias.clone()
    widen(p, 8)
    assert torch.equal(p.summary.n_head.weight[:9], w0[:9])
    assert torch.equal(p.summary.n_head.bias[:9], b0[:9])
    assert not torch.equal(p.summary.n_head.weight[9], w0[9])


def test_other_src():
    p = fresh()
    widen(p, 5)
    w, b = p.summary.n_head.weight, p.summary.n_head.bias
    for n in range(6, 17):
        assert torch.equal(w[n], w[5])
        assert torch.equal(b[n], b[5])


def test_nothing_else_changes():
    p = fresh()
    before = {k: v.clone() for k, v in p.state_dict().items()}
    widen(p, 8)
    for k, v in p.state_dict().items():
        if not k.startswith("summary.n_head"):
            assert torch.equal(v, before[k]), k


def test_big_n_logits_equal_src_logit():
    obs, _ = two_obs()
    p = fresh()
    widen(p, 8)
    logits = outputs_of(p, obs)[0]
    for n in range(9, 17):
        assert abs(float(logits[0, n] - logits[0, 8])) < 1e-5


def test_position_step_and_value_unchanged():
    obs, _ = two_obs()
    p = fresh()
    l0, v0 = outputs_of(p, obs)
    widen(p, 8)
    l1, v1 = outputs_of(p, obs)
    assert torch.equal(v0, v1)
    assert torch.equal(l0[1], l1[1])
    assert torch.equal(l0[0, :9], l1[0, :9])
    assert torch.equal(l0[0, 17:], l1[0, 17:])


def test_probabilities_unchanged_when_big_n_is_illegal():
    obs, masks = two_obs()
    assert not masks[0][9:17].any()
    p = fresh()
    m = torch.as_tensor(masks)
    lp0 = masked_log_probs(outputs_of(p, obs)[0], m)
    widen(p, 8)
    lp1 = masked_log_probs(outputs_of(p, obs)[0], m)
    assert torch.allclose(lp0, lp1, atol=1e-6)
    assert torch.equal(lp0.argmax(1), lp1.argmax(1))


def test_big_n_shares_probability_when_legal():
    obs, masks = two_obs()
    p = fresh()
    widen(p, 8)
    logits = outputs_of(p, obs)[0][:1]
    m = torch.zeros(1, masks.shape[1], dtype=torch.bool)
    m[0, 1:11] = True
    lp = masked_log_probs(logits, m)
    assert abs(float(lp[0, 9] - lp[0, 8])) < 1e-5
    assert abs(float(lp[0, 10] - lp[0, 8])) < 1e-5
    assert int(lp[0].argmax()) <= 8 or abs(float(lp[0].max() - lp[0, 8])) < 1e-5


def test_save_wide_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        a, b = str(Path(d) / "a.pt"), str(Path(d) / "b.pt")
        src = fresh(1)
        save_checkpoint(a, src, torch.optim.Adam(src.parameters()), 7, {"best_eval": 0.5})
        save_wide(a, 8, b)
        q = Policy()
        r, extra = load_checkpoint(b, q)
        assert r == 0
        assert extra["src"] == 8 and extra["widened_from"] == a
        want = fresh(1)
        widen(want, 8)
        for k, v in want.state_dict().items():
            assert torch.equal(v, q.state_dict()[k]), k
        again = Policy()
        load_checkpoint(a, again)
        assert torch.equal(again.summary.n_head.weight, src.summary.n_head.weight)
