import sys

from executor.bridge import Bridge
from executor.plan import CONNECTABLE
from executor.build import load_record, to_world

POS_TOL = 0.01  
YAW_TOL = 0.5  


def yaw_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def check(plan, origin, site_yaw, report):
    problems = []
    actual = {p["id"]: p for p in report["pieces"]}
    entities = {e["id"]: e for e in plan["entities"]}

    for eid, e in entities.items():
        a = actual.get(eid)
        if a is None:
            problems.append(f"{eid}: not tracked by the mod")
            continue
        if not a["exists"]:
            problems.append(f"{eid}: missing from the world (dismantled?)")
            continue
        if a["class"] != e["class"]:
            problems.append(f"{eid}: class is {a['class']}, plan says {e['class']}")
        if e["kind"] != "belt":
            want = to_world(e["pos"], origin, site_yaw)
            err = max(abs(w - g) for w, g in zip(want, a["pos"]))
            if err > POS_TOL:
                problems.append(f"{eid}: position off by {err * 100:.1f} cm")
            want_yaw = (e["yaw"] + site_yaw) % 360
            if yaw_diff(want_yaw, a["yaw"]) > YAW_TOL:
                problems.append(f"{eid}: yaw is {a['yaw']:.1f}, plan says {want_yaw:.1f}")

    for aid in actual:
        if aid not in entities:
            problems.append(f"{aid}: tracked by the mod but not in the plan")

    links_per_machine = {}
    for e in entities.values():
        if e["kind"] != "belt":
            continue
        for m in (e["from"], e["to"]):
            links_per_machine[m] = links_per_machine.get(m, 0) + 1
        a = actual.get(e["id"])
        if not a or not a["exists"]:
            continue

        ends = sorted(p["to"] for p in a["ports"] if p["connected"])
        if ends != sorted([e["from"], e["to"]]):
            problems.append(f"{e['id']}: connected to {ends or 'nothing'}, plan says {[e['from'], e['to']]}")
        for machine, want_dir, want_name in ((e["from"], "out", e.get("from_port")),
                                             (e["to"], "in", e.get("to_port"))):
            m = actual.get(machine)
            if not m or not m["exists"]:
                continue
            ports = [p for p in m["ports"] if p["to"] == e["id"]]
            if not ports or ports[0]["dir"] != want_dir:
                problems.append(f"{e['id']}: should join {machine}'s {want_dir}put port")
            elif want_name and ports[0]["name"] != want_name:
                problems.append(f"{e['id']}: joins {machine}.{ports[0]['name']}, plan says {want_name}")

    for eid, e in entities.items():
        a = actual.get(eid)
        if e["kind"] not in CONNECTABLE or not a or not a["exists"]:
            continue
        got = sum(1 for p in a["ports"] if p["connected"])
        want = links_per_machine.get(eid, 0)
        if got != want:
            problems.append(f"{eid}: {got} connected port(s), plan expects {want}")

    return problems


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m executor.verify <build_id>")
    build_id = sys.argv[1]

    record = load_record(build_id)
    report = Bridge().verify(build_id)
    problems = check(record["plan"], record["origin"], record["yaw"], report)

    n_pieces = len(record["plan"]["entities"])
    n_belts = sum(1 for e in record["plan"]["entities"] if e["kind"] == "belt")
    if problems:
        print(f"FAIL: {build_id!r}, {len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"PASS: {build_id!r}, {n_pieces} pieces in place, {n_belts} belts connected as planned")


if __name__ == "__main__":
    main()