import math
import re
from pathlib import Path

import torch

from rl.dist import NEG, masked_log_probs, sample, pick, entropy


def test_rl_imports_nothing_from_project():
    bad = re.compile(r"^\s*(from|import)\s+(layout|solver|siting|executor)\b")
    files = list(Path("rl").glob("*.py"))
    assert len(files) >= 2
    for f in files:
        for line in f.read_text().splitlines():
            assert not bad.match(line), f"{f}: {line}"


def test_pick_hand():
    log_p = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out = pick(log_p, torch.tensor([2, 0]))
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([0.3, 0.4]))


def test_log_probs_hand():
    logits = torch.tensor([[1.0, 2.0, 0.0, 3.0]])
    mask = torch.tensor([[True, True, True, False]])
    lp = masked_log_probs(logits, mask)
    assert lp.shape == (1, 4) and lp.dtype == torch.float32
    p = lp.exp()[0]
    assert torch.allclose(p, torch.tensor([0.2447, 0.6652, 0.0900, 0.0]), atol=1e-3)


def test_log_probs_rows_sum_to_one():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 10, generator=g) * 3
    mask = torch.rand(5, 10, generator=g) < 0.5
    mask[:, 7] = True
    p = masked_log_probs(logits, mask).exp()
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)
    assert (p[~mask] == 0).all()


def test_log_probs_illegal_tiny_not_inf():
    logits = torch.tensor([[1.0, 2.0, 0.0, 3.0]])
    mask = torch.tensor([[True, True, True, False]])
    lp = masked_log_probs(logits, mask)
    assert torch.isfinite(lp).all()
    assert lp[0, 3] < -1e8
    assert NEG == -1e9


def test_log_probs_does_not_change_input():
    logits = torch.tensor([[1.0, 2.0, 0.0, 3.0]])
    before = logits.clone()
    masked_log_probs(logits, torch.tensor([[True, False, True, False]]))
    assert torch.equal(logits, before)


def test_log_probs_gradient_hand():
    logits = torch.tensor([[1.0, 2.0, 0.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])
    lp = masked_log_probs(logits, mask)
    loss = -pick(lp, torch.tensor([2])).sum()
    loss.backward()
    assert not torch.isnan(logits.grad).any()
    assert torch.allclose(logits.grad[0], torch.tensor([0.2447, 0.6652, -0.9100, 0.0]), atol=1e-3)


def test_log_probs_batch_independent():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 6, generator=g)
    mask = torch.rand(3, 6, generator=g) < 0.6
    mask[:, 0] = True
    whole = masked_log_probs(logits, mask)
    for i in range(3):
        one = masked_log_probs(logits[i:i + 1], mask[i:i + 1])
        assert torch.allclose(whole[i], one[0])


def test_sample_shape_dtype():
    lp = masked_log_probs(torch.zeros(7, 5), torch.ones(7, 5, dtype=torch.bool))
    a = sample(lp, torch.Generator().manual_seed(0))
    assert a.shape == (7,) and a.dtype == torch.int64


def test_sample_only_legal():
    mask = torch.zeros(500, 6, dtype=torch.bool)
    mask[:, 1] = True
    mask[::2, 4] = True
    lp = masked_log_probs(torch.zeros(500, 6), mask)
    a = sample(lp, torch.Generator().manual_seed(0))
    assert mask[torch.arange(500), a].all()
    assert (a == 1).any() and (a == 4).any()


def test_sample_frequencies():
    logits = torch.tensor([[0.0, math.log(3)]]).repeat(4000, 1)
    lp = masked_log_probs(logits, torch.ones(4000, 2, dtype=torch.bool))
    a = sample(lp, torch.Generator().manual_seed(0))
    assert 0.72 < a.float().mean().item() < 0.78


def test_sample_repeatable():
    lp = masked_log_probs(torch.zeros(50, 9), torch.ones(50, 9, dtype=torch.bool))
    a = sample(lp, torch.Generator().manual_seed(3))
    b = sample(lp, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_entropy_hand():
    lp = masked_log_probs(torch.tensor([[1.0, 2.0, 0.0, 3.0]]), torch.tensor([[True, True, True, False]]))
    h = entropy(lp)
    assert h.shape == (1,)
    assert abs(h.item() - 0.8324) < 2e-3


def test_entropy_uniform():
    mask = torch.ones(2, 27648, dtype=torch.bool)
    mask[1, ::2] = False
    h = entropy(masked_log_probs(torch.zeros(2, 27648), mask))
    assert torch.allclose(h, torch.tensor([10.2273, 9.5342]), atol=1e-3)


def test_entropy_certain_is_zero():
    h = entropy(masked_log_probs(torch.tensor([[0.0, 50.0]]), torch.ones(1, 2, dtype=torch.bool)))
    assert 0 <= h.item() < 1e-5


def test_entropy_ignores_illegal():
    lp = masked_log_probs(torch.tensor([[0.0, 0.0, 7.0, -3.0]]), torch.tensor([[True, True, False, False]]))
    assert abs(entropy(lp).item() - math.log(2)) < 1e-5


def test_entropy_gradient_no_nan():
    logits = torch.tensor([[1.0, 2.0, 0.0, 3.0], [0.5, -1.0, 2.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [False, True, True, True]])
    entropy(masked_log_probs(logits, mask)).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~mask] == 0).all()
    assert (logits.grad[mask] != 0).any()


TESTS = [test_rl_imports_nothing_from_project, test_pick_hand, test_log_probs_hand, test_log_probs_rows_sum_to_one,
    test_log_probs_illegal_tiny_not_inf, test_log_probs_does_not_change_input, test_log_probs_gradient_hand,
    test_log_probs_batch_independent, test_sample_shape_dtype, test_sample_only_legal, test_sample_frequencies,
    test_sample_repeatable, test_entropy_hand, test_entropy_uniform, test_entropy_certain_is_zero,
    test_entropy_ignores_illegal, test_entropy_gradient_no_nan]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")