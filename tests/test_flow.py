from layout.connect import chains
from layout.view import random_episode
from layout.wire import links, route_all
from layout.flow import end_owner, chain_delivery, link_ok, one_pass, block_fracs, flow


def test_end_owner_seed0():
    env = random_episode(0)
    own = end_owner(env)
    total = sum(len(b["ends"]) for b in env.world.blocks) + len(env.entries) + 1
    assert len(own) == total
    for bid, b in enumerate(env.world.blocks):
        f = b["floor"]
        n_in = len(b["block"]["inputs"])
        for k in range(n_in):
            assert own[(f, *b["ends"][k])] == (bid, k)
        assert own[(f, *b["ends"][-1])] == (bid, "out")
    for item, (x, y) in env.entries.items():
        assert own[(0, x, y)] == ("raw", item)
    assert own[(0, *env.exit)] == ("exit", None)
    assert all(len(c) == 3 for c in own)


def test_end_owner_covers_chains():
    for seed in (4, 17):
        env = random_episode(seed)
        own = end_owner(env)
        for item, ch in chains(env).items():
            for c in ch["cells"]:
                assert c in own


def test_delivery_all_ok():
    assert chain_delivery([10, 20], [15, 10, 5], [True] * 4) == [15, 10, 5]
    assert chain_delivery([30], [30], [True]) == [30]


def test_delivery_producer_break():
    assert chain_delivery([10, 20], [15, 10, 5], [False, True, True, True]) == [15, 5, 0]
    assert chain_delivery([10, 20, 5], [35], [True, False, True]) == [5]
    assert chain_delivery([10, 20], [15, 15], [True, False, True]) == [0, 0]


def test_delivery_consumer_break():
    assert chain_delivery([10, 20], [15, 10, 5], [True, True, False, True]) == [15, 0, 0]
    assert chain_delivery([10, 20], [15, 10, 5], [True, True, True, False]) == [15, 10, 0]
    assert chain_delivery([30], [10, 10, 10], [True, False, True, True]) == [10, 0, 0]


def test_delivery_short_supply():
    assert chain_delivery([12], [15, 10], [True, True]) == [12, 0]
    assert chain_delivery([4, 6], [5, 5], [True, True, True]) == [5, 5]
    assert chain_delivery([0], [5], [True]) == [0]
    prod = [7, 3]
    need = [4, 4]
    ok = [True, True, True]
    chain_delivery(prod, need, ok)
    assert prod == [7, 3] and need == [4, 4] and ok == [True, True, True]


def test_link_ok():
    env = random_episode(4)
    res = route_all(env)
    ok = link_ok(env, res)
    ch = chains(env)
    assert set(ok) == set(ch)
    for item in ch:
        assert len(ok[item]) == len(ch[item]["cells"]) - 1
        assert all(ok[item])
    env = random_episode(26)
    res = route_all(env)
    ok = link_ok(env, res)
    assert sum(v.count(False) for v in ok.values()) == 1
    bad = res["failed"][0]
    mine = [L for L in links(env) if L[0] == bad[0]]
    assert ok[bad[0]] == [L != bad for L in mine]


def setup(env, failed=()):
    routes = {"failed": list(failed)}
    return chains(env), end_owner(env), link_ok(env, routes)


def link_of(env, item, k=0):
    return [L for L in links(env) if L[0] == item][k]


def test_one_pass_full():
    env = random_episode(0)
    ch, own, oks = setup(env)
    got = one_pass(env, ch, own, oks, [1.0, 1.0, 1.0])
    assert got == {(0, 0): 20, (1, 0): 20, (2, 0): 20, ("exit", None): 20}


def test_one_pass_scaled():
    env = random_episode(0)
    ch, own, oks = setup(env)
    got = one_pass(env, ch, own, oks, [0.5, 1.0, 1.0])
    assert got[(1, 0)] == 10
    assert got[(0, 0)] == 20 and got[(2, 0)] == 20 and got[("exit", None)] == 20
    got = one_pass(env, ch, own, oks, [1.0, 1.0, 0.0])
    assert got[("exit", None)] == 0


def test_block_fracs():
    env = random_episode(0)
    assert block_fracs(env, {(0, 0): 20, (1, 0): 5, (2, 0): 20, ("exit", None): 20}) == [1.0, 0.25, 1.0]
    assert block_fracs(env, {(0, 0): 30, (1, 0): 0, (2, 0): 20}) == [1.0, 0.0, 1.0]
    env = random_episode(4)
    b = env.world.blocks[0]["block"]
    r0, r1 = b["inputs"][0]["rate"], b["inputs"][1]["rate"]
    got = {(i, k): inp["rate"] for i, bb in enumerate(env.world.blocks) for k, inp in enumerate(bb["block"]["inputs"])}
    got[(0, 1)] = r1 / 2
    assert block_fracs(env, got)[0] == 0.5
    got[(0, 0)] = r0 / 4
    assert block_fracs(env, got)[0] == 0.25
    assert block_fracs(env, got)[1:] == [1.0] * (len(env.world.blocks) - 1)


def test_flow_all_routed_passes():
    for seed in range(15):
        env = random_episode(seed)
        if env.failed:
            continue
        res = flow(env, {"failed": []})
        assert res["passed"], seed
        assert abs(res["exit"] - res["target"]) < 1e-9
        assert abs(res["fraction"] - 1) < 1e-9
        assert res["short"] == []
        assert all(abs(f - 1) < 1e-9 for f in res["blocks"])


def test_flow_seed4_route_all():
    env = random_episode(4)
    res = flow(env, route_all(env))
    assert res["passed"] and res["fraction"] == 1 and res["target"] == env.graph["rate"]


def test_flow_ore_cut_ripples():
    env = random_episode(0)
    res = flow(env, {"failed": [link_of(env, "Desc_OreIron_C")]})
    assert res["exit"] == 0 and res["fraction"] == 0 and not res["passed"]
    assert res["blocks"] == [0.0, 0.0, 0.0]
    assert sorted(s[0] for s in res["short"]) == [0, 1, 2]
    assert ("Desc_OreIron_C") in [s[1] for s in res["short"]]


def test_flow_exit_cut():
    env = random_episode(0)
    res = flow(env, {"failed": [link_of(env, "Desc_SpikedRebar_C")]})
    assert res["blocks"] == [1.0, 1.0, 1.0]
    assert res["exit"] == 0 and not res["passed"]
    assert res["short"] == []


def test_flow_seed26_wire():
    env = random_episode(26)
    res = flow(env, {"failed": [link_of(env, "Desc_Wire_C", 0)]})
    assert res["fraction"] == 0
    assert sorted((s[0], s[1], s[2], s[3]) for s in res["short"]) == [(5, "Desc_Wire_C", 0, 30.0), (6, "Desc_Wire_C", 0, 30.0)]
    res = flow(env, {"failed": [link_of(env, "Desc_Wire_C", 1)]})
    assert abs(res["fraction"] - 0.5) < 1e-9
    assert abs(res["exit"] - 3.75) < 1e-9
    assert len(res["short"]) == 1 and res["short"][0][1] == "Desc_Wire_C"
    assert sorted(res["blocks"]) == [0.0] + [1.0] * 6


TESTS = [
    test_end_owner_seed0,
    test_end_owner_covers_chains,
    test_delivery_all_ok,
    test_delivery_producer_break,
    test_delivery_consumer_break,
    test_delivery_short_supply,
    test_link_ok,
    test_one_pass_full,
    test_one_pass_scaled,
    test_block_fracs,
    test_flow_all_routed_passes,
    test_flow_seed4_route_all,
    test_flow_ore_cut_ripples,
    test_flow_exit_cut,
    test_flow_seed26_wire,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")