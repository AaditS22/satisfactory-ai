"""Load and validate FactoryPlan files (schema: docs/factoryplan.md)."""
import json
from pathlib import Path

SCHEMA = "satai.factoryplan"
SUPPORTED_VERSIONS = {0}
KINDS = {"foundation", "machine", "belt"}


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_vec3(v):
    return (isinstance(v, list) and len(v) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def validate_plan(plan):
    """Return a list of problems. An empty list means the plan is valid."""
    problems = []

    if not isinstance(plan, dict):
        return ["plan is not a JSON object"]
    if plan.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}, got {plan.get('schema')!r}")
    if plan.get("version") not in SUPPORTED_VERSIONS:
        problems.append(f"unsupported version {plan.get('version')!r} (supported: {sorted(SUPPORTED_VERSIONS)})")

    entities = plan.get("entities")
    if not isinstance(entities, list) or not entities:
        problems.append("entities must be a non-empty list")
        return problems

    kinds_by_id = {}
    for i, e in enumerate(entities):
        where = f"entities[{i}]"
        if not isinstance(e, dict):
            problems.append(f"{where}: not an object")
            continue

        eid = e.get("id")
        if not isinstance(eid, str) or not eid:
            problems.append(f"{where}: missing or empty id")
        elif eid in kinds_by_id:
            problems.append(f"{where}: duplicate id {eid!r}")
        else:
            kinds_by_id[eid] = e.get("kind")
            where = f"{eid!r}"

        kind = e.get("kind")
        if kind not in KINDS:
            problems.append(f"{where}: kind must be one of {sorted(KINDS)}, got {kind!r}")
            continue

        if not isinstance(e.get("class"), str) or not e["class"].endswith("_C"):
            problems.append(f"{where}: class must be a string ending in '_C'")

        if kind in ("foundation", "machine"):
            if not _is_vec3(e.get("pos")):
                problems.append(f"{where}: pos must be [x, y, z] numbers")
            yaw = e.get("yaw")
            if not isinstance(yaw, (int, float)) or yaw % 90 != 0:
                problems.append(f"{where}: yaw must be a multiple of 90, got {yaw!r}")
        if kind == "machine" and not isinstance(e.get("recipe"), str):
            problems.append(f"{where}: machine needs a recipe")

    for e in entities:
        if not isinstance(e, dict) or e.get("kind") != "belt":
            continue
        for end in ("from", "to"):
            ref = e.get(end)
            if ref not in kinds_by_id:
                problems.append(f"{e.get('id')!r}: {end} refers to unknown id {ref!r}")
            elif kinds_by_id[ref] != "machine":
                problems.append(f"{e.get('id')!r}: {end} must be a machine, {ref!r} is a {kinds_by_id[ref]}")
        if e.get("from") is not None and e.get("from") == e.get("to"):
            problems.append(f"{e.get('id')!r}: from and to are the same entity")

    return problems


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/plans/iron_plate_2smelter.json")
    issues = validate_plan(load_plan(path))
    if issues:
        print(f"{path}: {len(issues)} problem(s)")
        for p in issues:
            print("  -", p)
        sys.exit(1)
    print(f"{path}: OK")