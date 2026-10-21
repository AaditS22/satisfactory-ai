import numpy as np
import torch

from layout.env import LayoutEnv
from layout import obs as O
from layout.policy import (flatten_floors, ConvBlock, GridEncoder, adjacency, neighbour_mean, masked_mean,
    head_rows, GraphEncoder, Summary, grow, add_to_every_cell, Decoder, stack_obs, obs_to_torch, merge_logits, Policy)
from layout.gym_env import observe


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def random_grid(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, 10, 3, 96, 96, generator=g) < 0.2).to(torch.uint8)


def test_flatten_floors():
    grid = torch.zeros(2, 10, 3, 96, 96, dtype=torch.uint8)
    grid[1, 4, 2, 7, 9] = 1
    grid[0, 0, 1, 95, 0] = 1
    x = flatten_floors(grid)
    assert x.shape == (2, 30, 96, 96)
    assert x.dtype == torch.float32
    assert x.sum().item() == 2
    assert x[1, 4 * 3 + 2, 7, 9].item() == 1
    assert x[0, 1, 95, 0].item() == 1


def test_conv_block_shape():
    torch.manual_seed(0)
    blk = ConvBlock(5, 7)
    y = blk(torch.randn(2, 5, 9, 11))
    assert y.shape == (2, 7, 9, 11)
    assert (y >= 0).all()
    assert y.max().item() > 0


def test_conv_block_params():
    assert n_params(ConvBlock(5, 7)) == 770
    assert n_params(ConvBlock(32, 32)) == 18496
    assert len(list(ConvBlock(5, 7).parameters())) == 4


def test_encoder_shapes():
    torch.manual_seed(0)
    enc = GridEncoder()
    out = enc(flatten_floors(random_grid(2)))
    assert isinstance(out, tuple) and len(out) == 4
    s1, s2, s3, bottom = out
    assert s1.shape == (2, 32, 48, 48)
    assert s2.shape == (2, 64, 24, 24)
    assert s3.shape == (2, 128, 12, 12)
    assert bottom.shape == (2, 128, 6, 6)
    for t in out:
        assert t.dtype == torch.float32
        assert (t >= 0).all()


def test_encoder_params():
    enc = GridEncoder()
    assert len(list(enc.parameters())) == 20
    assert n_params(enc) == 636128


def test_encoder_grad():
    torch.manual_seed(1)
    enc = GridEncoder()
    out = enc(flatten_floors(random_grid(2, seed=1)))
    loss = sum(t.mean() for t in out)
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_encoder_batch_independent():
    torch.manual_seed(2)
    enc = GridEncoder()
    grid = random_grid(3, seed=2)
    with torch.no_grad():
        both = enc(flatten_floors(grid))
        one = enc(flatten_floors(grid[1:2]))
    for a, b in zip(both, one):
        assert torch.allclose(a[1:2], b, atol=1e-5)
    assert not torch.allclose(both[3][0], both[3][1])


RIP = {"target": "Desc_IronPlateReinforced_C", "rate": 5, "width": 64, "depth": 64, "floors": 3, "side": 0}
PLATE = {"target": "Desc_IronPlate_C", "rate": 30, "width": 24, "depth": 24, "floors": 3, "side": 0}


def graph_batch():
    a = LayoutEnv()
    a.reset(dict(RIP))
    b = LayoutEnv()
    b.reset(dict(PLATE))
    b.step(2, 0, 0, 2, 2)
    gs = [O.graph_obs(a), O.graph_obs(b)]
    t = {k: torch.as_tensor(np.stack([g[k] for g in gs])) for k in gs[0]}
    return [t["nodes"], t["node_mask"], t["edges"], t["edge_rate"], t["edge_mask"]]


def small_edges():
    edges = torch.tensor([[[0, 1], [1, 2], [1, 3], [0, 0], [0, 0]]], dtype=torch.int32)
    mask = torch.tensor([[True, True, True, False, False]])
    return edges, mask


def test_adjacency_hand():
    edges, mask = small_edges()
    adj = adjacency(edges, mask, n=5)
    want = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.float32)
    assert adj.shape == (1, 5, 5)
    assert torch.equal(adj[0], want)
    assert adjacency(edges, torch.zeros(1, 5, dtype=torch.bool), n=5).sum().item() == 0


def test_neighbour_mean_rounds():
    edges, mask = small_edges()
    adj = adjacency(edges, mask, n=5)
    h = torch.tensor([[[0.0], [0.0], [0.0], [3.0], [0.0]]])
    h = h + neighbour_mean(h, adj)
    assert h[0, :, 0].tolist() == [0, 1, 0, 3, 0]
    h = h + neighbour_mean(h, adj)
    assert h[0, :, 0].tolist() == [1, 2, 1, 4, 0]
    assert torch.isfinite(h).all()


def test_masked_mean_hand():
    h = torch.tensor([[[2.0], [4.0], [6.0], [7.0], [9.0]]])
    mask = torch.tensor([[True, True, True, False, False]])
    assert masked_mean(h, mask).tolist() == [[4.0]]
    assert masked_mean(h, torch.zeros(1, 5, dtype=torch.bool)).tolist() == [[0.0]]


def test_head_rows_hand():
    h = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    nodes = torch.zeros(1, 5, O.N_NODE)
    assert head_rows(h, nodes).tolist() == [[0.0, 0.0]]
    nodes[0, 2, O.G_HEAD] = 1
    assert head_rows(h, nodes).tolist() == [[4.0, 5.0]]


def test_graph_encoder_shape():
    torch.manual_seed(0)
    enc = GraphEncoder()
    g = enc(*graph_batch())
    assert g.shape == (2, 64)
    assert g.dtype == torch.float32
    assert (g >= 0).all()


def test_graph_encoder_params():
    enc = GraphEncoder()
    assert len(list(enc.parameters())) == 8
    assert n_params(enc) == 25792


def test_graph_encoder_ignores_padding():
    torch.manual_seed(1)
    enc = GraphEncoder()
    nodes, node_mask, edges, rate, edge_mask = graph_batch()
    with torch.no_grad():
        before = enc(nodes, node_mask, edges, rate, edge_mask)
        junk = nodes.clone()
        pad = ~node_mask
        junk[pad] = torch.rand(int(pad.sum()), O.N_NODE)
        junk[:, :, O.G_HEAD] = nodes[:, :, O.G_HEAD]
        after = enc(junk, node_mask, edges, rate, edge_mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_graph_encoder_uses_edges():
    torch.manual_seed(2)
    enc = GraphEncoder()
    nodes, node_mask, edges, rate, edge_mask = graph_batch()
    with torch.no_grad():
        a = enc(nodes, node_mask, edges, rate, edge_mask)
        b = enc(nodes, node_mask, edges, rate, torch.zeros_like(edge_mask))
    assert not torch.allclose(a, b)


def test_graph_encoder_grad():
    torch.manual_seed(3)
    enc = GraphEncoder()
    enc(*graph_batch()).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_graph_encoder_batch_independent():
    torch.manual_seed(4)
    enc = GraphEncoder()
    batch = graph_batch()
    with torch.no_grad():
        both = enc(*batch)
        one = enc(*[t[1:2] for t in batch])
    assert torch.allclose(both[1:2], one, atol=1e-5)
    assert not torch.allclose(both[0], both[1])


def summary_inputs(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(b, 128, 6, 6, generator=g), torch.rand(b, 64, generator=g), torch.rand(b, 23, generator=g),
        torch.rand(b, 2, generator=g)]


def test_summary_shapes():
    torch.manual_seed(0)
    out = Summary()(*summary_inputs())
    assert isinstance(out, tuple) and len(out) == 3
    z, value, n_logits = out
    assert z.shape == (2, 128) and (z >= 0).all()
    assert value.shape == (2,)
    assert n_logits.shape == (2, 17)


def test_summary_params():
    s = Summary()
    assert len(list(s.parameters())) == 8
    assert n_params(s) == 46738


def test_summary_heads_can_be_negative():
    s = Summary()
    with torch.no_grad():
        s.value.weight.zero_()
        s.value.bias.fill_(-1.5)
        s.n_head.weight.zero_()
        s.n_head.bias.fill_(-3.0)
        z, value, n_logits = s(*summary_inputs())
    assert torch.allclose(value, torch.full((2,), -1.5))
    assert torch.allclose(n_logits, torch.full((2, 17), -3.0))


def test_summary_averages_the_map():
    torch.manual_seed(1)
    s = Summary()
    bottom, graph, vec, stage = summary_inputs(seed=1)
    with torch.no_grad():
        a = s(bottom, graph, vec, stage)
        b = s(torch.flip(bottom, dims=(2, 3)), graph, vec, stage)
        c = s(bottom.mean(dim=(2, 3), keepdim=True).expand(-1, -1, 6, 6), graph, vec, stage)
    for x, y, w in zip(a, b, c):
        assert torch.allclose(x, y, atol=1e-5)
        assert torch.allclose(x, w, atol=1e-5)


def test_summary_uses_every_input():
    torch.manual_seed(2)
    s = Summary()
    base = summary_inputs(seed=2)
    with torch.no_grad():
        z0 = s(*base)[0]
        for i in range(4):
            changed = [t.clone() for t in base]
            changed[i] = changed[i] + 1.0
            assert not torch.allclose(s(*changed)[0], z0), i


def test_summary_grad():
    torch.manual_seed(3)
    s = Summary()
    z, value, n_logits = s(*summary_inputs(seed=3))
    (z.sum() + value.sum() + n_logits.sum()).backward()
    for name, p in s.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def decoder_inputs(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(b, 32, 48, 48, generator=g), torch.rand(b, 64, 24, 24, generator=g),
        torch.rand(b, 128, 12, 12, generator=g), torch.rand(b, 128, 6, 6, generator=g), torch.rand(b, 128, generator=g)]


def test_grow_hand():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    want = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    assert grow(x)[0, 0].tolist() == want
    assert grow(torch.zeros(2, 5, 6, 6)).shape == (2, 5, 12, 12)


def test_add_to_every_cell_hand():
    x = torch.zeros(1, 2, 3, 3)
    v = torch.tensor([[5.0, -1.0]])
    y = add_to_every_cell(x, v)
    assert y.shape == (1, 2, 3, 3)
    assert (y[0, 0] == 5).all() and (y[0, 1] == -1).all()


def test_decoder_shape():
    torch.manual_seed(0)
    out = Decoder()(*decoder_inputs())
    assert out.shape == (2, 27648)
    assert out.dtype == torch.float32


def test_decoder_params():
    d = Decoder()
    assert len(list(d.parameters())) == 16
    assert n_params(d) == 644044


def test_decoder_map_order():
    d = Decoder()
    with torch.no_grad():
        d.head.weight.zero_()
        d.head.bias.copy_(torch.arange(12, dtype=torch.float32) - 5)
        out = d(*decoder_inputs())
    maps = out.reshape(2, 4, 3, 48, 48)
    for rot in range(4):
        for f in range(3):
            assert (maps[:, rot, f] == rot * 3 + f - 5).all(), (rot, f)


def test_decoder_uses_every_input():
    torch.manual_seed(1)
    d = Decoder()
    base = decoder_inputs(seed=1)
    with torch.no_grad():
        out0 = d(*base)
        for i in range(5):
            changed = [t.clone() for t in base]
            changed[i] = changed[i] + 1.0
            assert not torch.allclose(d(*changed), out0), i


def test_decoder_skip_is_local():
    torch.manual_seed(2)
    d = Decoder()
    base = decoder_inputs(seed=2)
    with torch.no_grad():
        a = d(*base).reshape(2, 12, 48, 48)
        poked = [t.clone() for t in base]
        poked[0][:, :, 0, 0] += 5.0
        b = d(*poked).reshape(2, 12, 48, 48)
    assert not torch.allclose(a[:, :, 0, 0], b[:, :, 0, 0])
    assert torch.allclose(a[:, :, 10:, 10:], b[:, :, 10:, 10:])


def test_decoder_grad():
    torch.manual_seed(3)
    d = Decoder()
    d(*decoder_inputs(seed=3)).sum().backward()
    for name, p in d.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def policy_obs(device="cpu"):
    a = LayoutEnv()
    a.reset(dict(RIP))
    b = LayoutEnv()
    b.reset(dict(PLATE))
    b.step(2, 0, 0, 2, 2)
    return obs_to_torch(stack_obs([observe(a, 0, 0), observe(b, 1, 1)]), device)


def test_obs_to_torch_types():
    obs = policy_obs()
    assert obs["grid"].dtype == torch.uint8 and obs["grid"].shape == (2, 10, 3, 96, 96)
    assert obs["edges"].dtype == torch.int64
    assert obs["node_mask"].dtype == torch.bool and obs["edge_mask"].dtype == torch.bool
    for k in ("vec", "stage", "nodes", "edge_rate"):
        assert obs[k].dtype == torch.float32, k
    assert obs["stage"][:, 0].tolist() == [0.0, 1.0]


def test_merge_logits_hand():
    n = torch.arange(17, dtype=torch.float32).repeat(2, 1) + 100
    pos = torch.zeros(2, 20)
    stage = torch.tensor([[0.0, 0.0], [1.0, 0.5]])
    out = merge_logits(n, pos, stage)
    assert out.shape == (2, 20)
    assert out[0, :17].tolist() == list(range(100, 117)) and out[0, 17:].tolist() == [0, 0, 0]
    assert out[1].tolist() == [0.0] * 20


def test_policy_shapes():
    torch.manual_seed(0)
    logits, value = Policy()(policy_obs())
    assert logits.shape == (2, 27648) and logits.dtype == torch.float32
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_policy_params():
    net = Policy()
    assert len(list(net.parameters())) == 52
    assert n_params(net) == 1352702
    assert n_params(net.grid) == 636128 and n_params(net.decoder) == 644044


def test_policy_stage_routing():
    net = Policy()
    with torch.no_grad():
        net.summary.n_head.weight.zero_()
        net.summary.n_head.bias.fill_(100.0)
        net.decoder.head.weight.zero_()
        net.decoder.head.bias.fill_(-7.0)
        logits, value = net(policy_obs())
    assert (logits[0, :17] == 100).all()
    assert (logits[0, 17:] == -7).all()
    assert (logits[1] == -7).all()


def test_policy_grad():
    torch.manual_seed(1)
    net = Policy()
    logits, value = net(policy_obs())
    (logits.sum() + value.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_policy_gpu():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    net = Policy()
    with torch.no_grad():
        a, va = net(policy_obs())
        net = net.to("cuda")
        b, vb = net(policy_obs("cuda"))
    assert b.device.type == "cuda"
    assert torch.allclose(a, b.cpu(), atol=1e-2, rtol=1e-2) and torch.allclose(va, vb.cpu(), atol=1e-2, rtol=1e-2)


TESTS = [test_flatten_floors, test_conv_block_shape, test_conv_block_params, test_encoder_shapes,
    test_encoder_params, test_encoder_grad, test_encoder_batch_independent, test_adjacency_hand,
    test_neighbour_mean_rounds, test_masked_mean_hand, test_head_rows_hand, test_graph_encoder_shape,
    test_graph_encoder_params, test_graph_encoder_ignores_padding, test_graph_encoder_uses_edges,
    test_graph_encoder_grad, test_graph_encoder_batch_independent, test_summary_shapes, test_summary_params,
    test_summary_heads_can_be_negative, test_summary_averages_the_map, test_summary_uses_every_input,
    test_summary_grad, test_grow_hand, test_add_to_every_cell_hand, test_decoder_shape, test_decoder_params,
    test_decoder_map_order, test_decoder_uses_every_input, test_decoder_skip_is_local, test_decoder_grad,
    test_obs_to_torch_types, test_merge_logits_hand, test_policy_shapes, test_policy_params, test_policy_stage_routing,
    test_policy_grad, test_policy_gpu]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")