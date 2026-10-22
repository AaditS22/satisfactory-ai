import shutil
from pathlib import Path

import torch
from torch import nn

from rl.checkpoint import save_checkpoint, load_checkpoint

TMP = Path("tests/_tmp_ckpt")


class Tiny(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.body = nn.Linear(3, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(torch.relu(self.body(x)))


def trained(seed=0):
    torch.manual_seed(seed)
    net = Tiny()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        net(torch.randn(5, 3)).square().sum().backward()
        opt.step()
    return net, opt


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def fresh():
    shutil.rmtree(TMP, ignore_errors=True)


def test_save_makes_folders_and_file():
    fresh()
    net, opt = trained()
    save_checkpoint(TMP / "a" / "b" / "c.pt", net, opt, 7)
    assert (TMP / "a" / "b" / "c.pt").exists()
    fresh()


def test_file_contents():
    fresh()
    net, opt = trained()
    save_checkpoint(TMP / "c.pt", net, opt, 7, {"best_eval": 0.61})
    data = torch.load(TMP / "c.pt")
    assert set(data) == {"policy", "optimizer", "round", "extra"}
    assert data["round"] == 7 and data["extra"] == {"best_eval": 0.61}
    assert set(data["policy"]) == {"body.weight", "body.bias", "head.weight", "head.bias"}
    fresh()


def test_extra_defaults_to_empty_dict():
    fresh()
    net, opt = trained()
    save_checkpoint(TMP / "c.pt", net, opt, 0)
    assert load_checkpoint(TMP / "c.pt", Tiny()) == (0, {})
    fresh()


def test_load_restores_weights_and_returns_round():
    fresh()
    net, opt = trained(0)
    save_checkpoint(TMP / "c.pt", net, opt, 12, {"note": "x"})
    other = Tiny()
    assert not same(net, other)
    r, extra = load_checkpoint(TMP / "c.pt", other)
    assert r == 12 and extra == {"note": "x"} and same(net, other)
    fresh()


def test_load_restores_optimizer_memory():
    fresh()
    net, opt = trained(0)
    save_checkpoint(TMP / "c.pt", net, opt, 3)
    net2 = Tiny()
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-2)
    load_checkpoint(TMP / "c.pt", net2, opt2)
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert set(s1) == set(s2) and len(s2) == 4
    for k in s1:
        assert torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) and torch.equal(s1[k]["step"], s2[k]["step"])
    fresh()


def test_continuing_matches_never_stopping():
    fresh()
    net, opt = trained(0)
    save_checkpoint(TMP / "c.pt", net, opt, 3)
    net2 = Tiny()
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-2)
    load_checkpoint(TMP / "c.pt", net2, opt2)
    x = torch.randn(5, 3)
    for n, o in ((net, opt), (net2, opt2)):
        o.zero_grad()
        n(x).square().sum().backward()
        o.step()
    assert same(net, net2)
    fresh()


def test_wrong_network_raises():
    fresh()
    net, opt = trained()
    save_checkpoint(TMP / "c.pt", net, opt, 1)
    try:
        load_checkpoint(TMP / "c.pt", Tiny(hidden=5))
        assert False
    except RuntimeError:
        pass
    fresh()


def test_load_onto_device():
    fresh()
    net, opt = trained()
    save_checkpoint(TMP / "c.pt", net, opt, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    other = Tiny().to(device)
    load_checkpoint(TMP / "c.pt", other, device=device)
    assert next(other.parameters()).device.type == device
    assert torch.equal(other.body.weight.cpu(), net.body.weight)
    fresh()


TESTS = [test_save_makes_folders_and_file, test_file_contents, test_extra_defaults_to_empty_dict,
    test_load_restores_weights_and_returns_round, test_load_restores_optimizer_memory,
    test_continuing_matches_never_stopping, test_wrong_network_raises, test_load_onto_device]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")