
import math
import sys

from executor.bridge import Bridge
from executor.paths import built_with_path, class_path


def describe_origin_height(lo_z, hi_z, tol=0.05):
    if abs(hi_z) < tol:
        return "origin is on the TOP surface"
    if abs(lo_z) < tol:
        return "origin is on the BOTTOM surface"
    if abs(lo_z + hi_z) < tol:
        return "origin is at the vertical CENTRE"
    return f"origin is {-lo_z:.2f} m above the bottom"


def fmt(v):
    return [round(x, 2) for x in v]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Build_Foundation_8x1_01_C"
    bridge = Bridge()

    player = bridge.player()
    raw_yaw = player["yaw"] % 360
    yaw = round(raw_yaw / 90) * 90 % 360
    px, py, _ = player["pos"]
    x = px + 15 * math.cos(math.radians(yaw))
    y = py + 15 * math.sin(math.radians(yaw))
    z = bridge.ground(x, y)
    print(f"you were looking at yaw {raw_yaw:.0f}, snapped to {yaw}")

    r = bridge.spawn(class_path(name), (x, y, z), yaw, build_id="probe", built_with=built_with_path(name))
    print(f"spawned {r['name']} at {fmt(r['pos'])} yaw {r['yaw']:.0f}  (lightweight: {r['lightweight']})")

    lo, hi = r["bounds_min"], r["bounds_max"]
    if lo is None:
        print("no bounds available")
    else:
        print(f"building-local bounds: min {fmt(lo)}  max {fmt(hi)}")
        print(f"size (x, y, z): {fmt(hi[i] - lo[i] for i in range(3))} m")
        print(describe_origin_height(lo[2], hi[2]))

    input("\nLook at it in-game, then press Enter here to remove it...")
    print(f"removed {bridge.clear('probe')} piece(s)")


if __name__ == "__main__":
    main()