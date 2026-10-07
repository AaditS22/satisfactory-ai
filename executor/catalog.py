import json
import math
from pathlib import Path

from executor.bridge import Bridge, BridgeError
from executor.paths import class_path

CATALOG_BUILDINGS = [
    "Build_SmelterMk1_C",
    "Build_ConstructorMk1_C",
    "Build_AssemblerMk1_C",
    "Build_ManufacturerMk1_C",
    "Build_FoundryMk1_C",
    "Build_ConveyorAttachmentSplitter_C",
    "Build_ConveyorAttachmentMerger_C",
    "Build_StorageContainerMk1_C",
    "Build_Foundation_8x1_01_C",
    "Build_Foundation_8x2_01_C",
    "Build_Foundation_8x4_01_C",
    "Build_ConveyorLiftMk1_C",
    "Build_MinerMk1_C",
    "Build_MinerMk2_C",
    "Build_MinerMk3_C",
]

OUT = Path("data/catalog.json")
HEIGHT_ABOVE_GROUND = 30.0 


def r3(v):
    return [round(x, 3) + 0.0 for x in v] 

def union_box(boxes):
    if not boxes:
        return None, None
    lo = [min(b["min"][i] for b in boxes) for i in range(3)]
    hi = [max(b["max"][i] for b in boxes) for i in range(3)]
    return lo, hi

def main():
    bridge = Bridge()
    player = bridge.player()
    yaw = math.radians(player["yaw"])
    x = player["pos"][0] + 20 * math.cos(yaw)
    y = player["pos"][1] + 20 * math.sin(yaw)
    z = bridge.ground(x, y) + HEIGHT_ABOVE_GROUND

    buildings, failed = {}, {}
    for name in CATALOG_BUILDINGS:
        try:
            r = bridge.spawn(class_path(name), (x, y, z), 0, build_id="catalog")
        except (KeyError, BridgeError) as e:
            failed[name] = str(e)
            print(f"FAIL {name}: {e}")
            continue
        finally:
            bridge.clear("catalog")

        lo, hi = r["bounds_min"], r["bounds_max"]
        ports = [{"name": p["name"], "dir": p["dir"], "pos": r3(p["pos"]), "facing": r3(p["facing"])}
                 for p in r.get("ports", [])]
        clearance = [{"type": c["type"], "min": r3(c["min"]), "max": r3(c["max"])}
                     for c in r.get("clearance", [])]
        hard = [c for c in clearance if c["type"] != "soft"]
        c_lo, c_hi = union_box(hard or clearance)
        buildings[name] = {
            "lightweight": r["lightweight"],
            "bounds_min": r3(lo) if lo else None,
            "bounds_max": r3(hi) if hi else None,
            "size": r3([hi[i] - lo[i] for i in range(3)]) if lo else None,
            "clearance": clearance,
            "clearance_min": c_lo,
            "clearance_max": c_hi,
            "ports": ports,
        }
        n_in = sum(p["dir"] == "in" for p in ports)
        n_out = sum(p["dir"] == "out" for p in ports)
        n_any = len(ports) - n_in - n_out
        c_size = r3([c_hi[i] - c_lo[i] for i in range(3)]) if c_lo else "NONE"
        print(f"ok   {name:38} size {buildings[name]['size']}  clearance {c_size} ({len(clearance)} box)  "
              f"ports in={n_in} out={n_out} any={n_any}")

    OUT.write_text(json.dumps({
        "schema": "satai.catalog",
        "version": 1,
        "source": "executor.catalog: each building spawned in-game at yaw 0, building-local metres",
        "buildings": buildings,
        "failed": failed,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}: {len(buildings)} buildings, {len(failed)} failed")


if __name__ == "__main__":
    main()