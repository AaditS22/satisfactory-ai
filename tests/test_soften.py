import tempfile
from pathlib import Path

import numpy as np
import torch

from layout.soften import soften, load_soft, save_soft, entropy_chooser, pick_T
from layout.policy import Policy, obs_to_torch
from layout.gym_env import LayoutGym
from layout.evaluate import load_eval
from rl.checkpoint import save_checkpoint, load_checkpoint
from rl.collect import stack
from rl.dist import masked_log_probs, entropy


def two_obs():
    gym = LayoutGym()
    o0, _ = gym.reset(options={"task": load_eval(1)[0]["task"]})
    m0 = gym.action_masks()
    o1, _, _, _, _ = gym.step(int(np.flatnonzero(m0)[-1]))
    m1 = gym.action_masks()
    return [o0, o1], np.stack([m0, m1])


def logits_of(policy, obs_list):
    with torch.no_grad():
        return policy(obs_to_torch(stack(obs_list)))[0]


def fresh(seed=0):
    torch.manual_seed(seed)
    return Policy().eval()


def test_logits_divided_by_T():
    obs, _ = two_obs()
    p = fresh()
    before = logits_of(p, obs)
    soften(p, 2.0)
    after = logits_of(p, obs)
    assert torch.allclose(after, before / 2.0, atol=1e-5)


def test_T_one_changes_nothing():
    obs, _ = two_obs()
    p = fresh()
    before = logits_of(p, obs)
    soften(p, 1.0)
    assert torch.allclose(logits_of(p, obs), before, atol=1e-6)


def test_greedy_move_unchanged_and_entropy_up():
    obs, masks = two_obs()
    p = fresh()
    m = torch.as_tensor(masks)
    lp0 = masked_log_probs(logits_of(p, obs), m)
    soften(p, 3.0)
    lp1 = masked_log_probs(logits_of(p, obs), m)
    assert torch.equal(lp0.argmax(1), lp1.argmax(1))
    assert bool((entropy(lp1) > entropy(lp0)).all())


def test_only_output_layers_change():
    p = fresh()
    before = {k: v.clone() for k, v in p.state_dict().items()}
    soften(p, 2.0)
    changed = {k for k, v in p.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {"summary.n_head.weight", "summary.n_head.bias", "decoder.head.weight", "decoder.head.bias"}


def test_value_unchanged():
    obs, _ = two_obs()
    p = fresh()
    with torch.no_grad():
        v0 = p(obs_to_torch(stack(obs)))[1]
        soften(p, 2.0)
        v1 = p(obs_to_torch(stack(obs)))[1]
    assert torch.equal(v0, v1)


def test_save_soft_roundtrip():
    obs, _ = two_obs()
    p = fresh(1)
    before = logits_of(p, obs)
    with tempfile.TemporaryDirectory() as d:
        src, dst = str(Path(d) / "a.pt"), str(Path(d) / "b.pt")
        save_checkpoint(src, p, torch.optim.Adam(p.parameters()), 5, {})
        save_soft(src, 2.5, dst, "cpu")
        q = Policy()
        r, extra = load_checkpoint(dst, q)
        q.eval()
        assert r == 0 and extra["T"] == 2.5 and extra["softened_from"] == src
        assert torch.allclose(logits_of(q, obs), before / 2.5, atol=1e-5)
        same = Policy()
        load_checkpoint(src, same)
        assert torch.equal(same.decoder.head.weight, p.decoder.head.weight)


def test_load_soft_matches_soften():
    obs, _ = two_obs()
    p = fresh(2)
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "a.pt")
        save_checkpoint(src, p, torch.optim.Adam(p.parameters()), 0, {})
        q = load_soft(src, 1.5, "cpu")
    soften(p, 1.5)
    assert torch.allclose(logits_of(q, obs), logits_of(p, obs), atol=1e-6)


def test_entropy_chooser_records_and_is_legal():
    obs, masks = two_obs()
    record = []
    choose = entropy_chooser(fresh(), torch.Generator().manual_seed(0), record)
    a = choose(obs, masks)
    assert len(record) == 2 and all(e >= 0 for e in record)
    assert masks[0][a[0]] and masks[1][a[1]]


def test_pick_T():
    res = {1.0: 0.6, 2.0: 1.3, 1.5: 1.0, 3.0: 1.8}
    assert pick_T(res, 1.2) == 2.0
    assert pick_T(res, 1.7) == 3.0
    assert pick_T(res, 0.5) == 1.0
    assert pick_T(res, 5.0) == 3.0


TESTS = [test_pick_T, test_logits_divided_by_T, test_T_one_changes_nothing, test_greedy_move_unchanged_and_entropy_up,
    test_only_output_layers_change, test_value_unchanged, test_save_soft_roundtrip, test_load_soft_matches_soften,
    test_entropy_chooser_records_and_is_legal]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(TESTS)} passed")