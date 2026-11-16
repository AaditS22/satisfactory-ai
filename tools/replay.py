from layout.api import evaluate

TERMS = ["length", "lifts", "crossings", "floors", "bbox", "used", "penalty", "p_belt", "p_lifts", "p_crossings",
         "p_sprawl", "p_floors", "fraction", "routed", "placed"]

def kind_of(terms):
    """What happened in a finished game: failed, invalid or valid"""
    if "passed" not in terms:
        return "failed"
    return "valid" if terms["passed"] else "invalid"

def replay(task, plan):
    """Scores a plan with the real scorer and keeps the score, the kind and the penalty terms"""
    out = {"plan": plan, "blocks": [m[0] for m in plan]}
    try:
        res = evaluate(task, plan)
    except Exception as e:
        return {**out, "score": -2.0, "kind": "error", "error": f"{type(e).__name__}: {e}", "terms": {}}
    terms = res["terms"]
    return {**out, "score": float(res["score"]), "kind": kind_of(terms),
            "terms": {k: float(terms[k]) for k in TERMS if k in terms}}

def broken(result):
    """A layout that does not work (it scores <= 0)"""
    return result["kind"] != "valid"
