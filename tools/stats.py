import math

def mean(xs):
    """Mean of a list, or None if it is empty"""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None

def verdict(diff, se, tie=0.005):
    """The pre-registered rule: helps / neutral / worse / inconclusive"""
    if diff is None:
        return None
    clear = se is not None and abs(diff) > 2 * se
    if diff >= tie and clear:
        return "helps"
    if diff <= -tie and clear:
        return "worse"
    if abs(diff) <= tie:
        return "neutral"
    return "inconclusive"

def paired(a, b, tie=0.005):
    """Compares two score lists task by task: mean difference a - b, its SE and the verdict"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n == 0:
        return {"n": 0, "diff": None, "se": None, "n_se": None, "better": 0, "worse": 0, "verdict": None}
    diff = sum(d) / n
    se = math.sqrt(sum((x - diff) ** 2 for x in d) / (n - 1) / n) if n > 1 else None
    n_se = abs(diff) / se if se else None
    return {"n": n, "diff": diff, "se": se, "n_se": n_se, "better": sum(x > 0 for x in d),
            "worse": sum(x < 0 for x in d), "verdict": verdict(diff, se, tie)}

def band(rate, edges):
    """Name of the rate band a task falls in, e.g. '<5', '5-14', '>=15'"""
    lo = None
    for e in edges:
        if rate < e:
            return f"<{e}" if lo is None else f"{lo}-{e - 1}"
        lo = e
    return f">={lo}"

def bands(edges):
    """All band names in order"""
    return [band(e - 1, edges) for e in edges] + [band(edges[-1], edges)]
