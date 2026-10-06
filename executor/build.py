import argparse
import math
import sys

from executor.bridge import Bridge
from executor.paths import built_with_path, class_path, recipe_path
from executor.plan import load_plan, validate_plan

KIND_ORDER = {"foundation": 0, "machine": 1, "belt": 2} 
FOUNDATION_HALF = 4.0 
FLOOR_CLEARANCE = 0.2 


def rotate(x, y, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return x * c - y * s, x * s + y * c


def to_world(pos, origin, site_yaw):
    x, y = rotate(pos[0], pos[1], site_yaw)
    return [origin[0] + x, origin[1] + y, origin[2] + pos[2]]


def choose_site(bridge, plan, distance=10.0):
    player = bridge.player()
    yaw = round(player["yaw"] / 90) * 90 % 360
    px, py, _ = player["pos"]
    fx, fy = rotate(1, 0, yaw)

    foundations = [e for e in plan["entities"] if e["kind"] == "foundation"]
    min_x = min(e["pos"][0] for e in foundations) - FOUNDATION_HALF
    min_y = min(e["pos"][1] for e in foundations) - FOUNDATION_HALF
    max_y = max(e["pos"][1] for e in foundations) + FOUNDATION_HALF

    rx, ry = rotate(min_x, (min_y + max_y) / 2, yaw)
    ox, oy = px + fx * distance - rx, py + fy * distance - ry

    heights = []
    for e in foundations:
        for dx in (-FOUNDATION_HALF, 0, FOUNDATION_HALF):
            for dy in (-FOUNDATION_HALF, 0, FOUNDATION_HALF):
                wx, wy, _ = to_world([e["pos"][0] + dx, e["pos"][1] + dy, 0], [ox, oy, 0], yaw)
                heights.append(bridge.ground(wx, wy))
    print(f"ground under site: {min(heights):.2f} to {max(heights):.2f} m")
    return [ox, oy, max(heights) + FLOOR_CLEARANCE], yaw


def resolve(plan, origin, site_yaw):
    pieces = []
    for e in sorted(plan["entities"], key=lambda e: KIND_ORDER[e["kind"]]):
        p = {"id": e["id"], "kind": e["kind"], "class": class_path(e["class"])}
        built_with = built_with_path(e["class"])
        if built_with:
            p["built_with"] = built_with
        if e["kind"] == "belt":
            p["from"], p["to"] = e["from"], e["to"]
        else:
            p["pos"] = to_world(e["pos"], origin, site_yaw)
            p["yaw"] = (e["yaw"] + site_yaw) % 360
            if e["kind"] == "machine":
                p["recipe"] = recipe_path(e["recipe"])
        pieces.append(p)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--clear", metavar="BUILD_ID")
    args = ap.parse_args()
    bridge = Bridge()

    if args.clear:
        print(f"removed {bridge.clear(args.clear)} piece(s) from {args.clear!r}")
        return
    if not args.plan:
        ap.error("give a plan file, or --clear BUILD_ID")

    plan = load_plan(args.plan)
    problems = validate_plan(plan)
    if problems:
        print("plan is invalid:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    origin, yaw = choose_site(bridge, plan)
    build_id = plan["name"]
    result = bridge.build(build_id, resolve(plan, origin, yaw))
    print(f"built {result['built']} pieces as {build_id!r}, origin {[round(v, 2) for v in origin]}, yaw {yaw}")
    print(f"undo with:  python -m executor.build --clear {build_id}")


if __name__ == "__main__":
    main()