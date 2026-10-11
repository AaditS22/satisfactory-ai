from layout.wire import links
from layout.connect import chains

EPS = 1e-9
MAX_PASSES = 50

def end_owner(env):
    """Finds the owner for every end cell of a finished episode"""
    out = {}

    for bid, b in enumerate(env.world.blocks):
        f = b["floor"]
        n_in = len(b["block"]["inputs"])

        for k in range(n_in):
            out[(f, *b["ends"][k])] = (bid, k)

        out[(f, *b["ends"][-1])] = (bid, "out")

    for item, cell in env.entries.items():
        out[(0, *cell)] = ("raw", item)

    out[(0, *env.exit)] = ("exit", None)

    return out

def chain_delivery(prod, need, ok):
    """Finds how much each consumer of one chain receives"""
    P = len(prod)
    supply = 0

    for i in range(P):
        if i > 0 and not ok[i - 1]:
            supply = 0

        supply += prod[i]

    got = []
    for j in range(len(need)):
        if not ok[P - 1 + j]:
            supply = 0

        take = min(need[j], supply)
        got.append(take)
        supply -= take

    return got

def link_ok(env, routes):
    """Checks which links were routed and which failed"""
    failed = set(routes["failed"])
    out = {}

    for L in links(env):
        out.setdefault(L[0], []).append(L not in failed)

    return out

def one_pass(env, ch, own, oks, frac):
    """One sweep over every chain with the blocks running at the given fractions"""
    got = {}
    for item, c in ch.items():
        cells = c["cells"]
        P = c["n_prod"]

        need = []
        for cell in cells[P:]:
            o = own[cell]
            if o[0] == "exit":
                need.append(env.graph["rate"])
            else:
                need.append(env.world.blocks[o[0]]["block"]["inputs"][o[1]]["rate"])

        prod = []
        for cell in cells[:P]:
            o = own[cell]
            if o[0] == "raw":
                prod.append(sum(need))
            else:
                prod.append(env.world.blocks[o[0]]["block"]["output"]["rate"] * frac[o[0]])

        recv = chain_delivery(prod, need, oks[item])
        for cell, r in zip(cells[P:], recv):
            got[own[cell]] = r
    return got

def block_fracs(env, got):
    """Find how hard each block can run given what its inputs received"""
    out = []
    for bid, b in enumerate(env.world.blocks):
        f = 1.0
        for k, inp in enumerate(b["block"]["inputs"]):
            f = min(f, got[(bid, k)] / inp["rate"])
        out.append(f)
    return out

def flow(env, routes):
    """Conducts a full flow check for a routed episode"""
    ch = chains(env)
    own = end_owner(env)
    oks = link_ok(env, routes)
    frac = [1.0] * len(env.world.blocks)

    for _ in range(MAX_PASSES):
        got = one_pass(env, ch, own, oks, frac)
        new = block_fracs(env, got)
        if new == frac:
            break
        frac = new

    target = env.graph["rate"]
    exit_rate = got[("exit", None)]

    short = []
    for bid, b in enumerate(env.world.blocks):
        for k, inp in enumerate(b["block"]["inputs"]):
            if got[(bid, k)] < inp["rate"] - EPS:
                short.append((bid, inp["item"], got[(bid, k)], inp["rate"]))

    passed = exit_rate >= target - EPS and not short
    return {"exit": exit_rate, "target": target, "fraction": exit_rate / target,
        "passed": passed, "blocks": frac, "short": short}