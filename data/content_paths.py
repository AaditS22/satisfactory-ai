import json
import re

DESC_RE = re.compile(r"(/Game/[^'\"]+?)/Desc_(\w+?)\.Desc_\w+?_C")
FULLNAME_RE = re.compile(r"(/Game/\S+?_C)\b")


def native_name(group):
    return group["NativeClass"].split(".")[-1].strip("'\"")


def main():
    with open("data/raw/en-US.json", encoding="utf-16") as f:
        docs = json.load(f)
    recipe_group = next(g for g in docs if native_name(g) == "FGRecipe")

    buildings, built_with, recipe_paths = {}, {}, {}
    for c in recipe_group["Classes"]:
        m = FULLNAME_RE.search(c.get("FullName", ""))
        if m:
            recipe_paths[c["ClassName"]] = m.group(1)

        if "BuildGun" not in c.get("mProducedIn", ""):
            continue
        m = DESC_RE.search(c.get("mProduct", ""))
        if not m:
            continue
        folder, suffix = m.groups()
        name = f"Build_{suffix}_C"
        buildings[name] = f"{folder}/Build_{suffix}.Build_{suffix}_C"
        built_with[name] = c["ClassName"]

    out = {"buildings": buildings, "built_with": built_with, "recipes": recipe_paths}
    with open("data/content_paths.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"{len(buildings)} building paths, {len(recipe_paths)} recipe paths -> data/content_paths.json")
    for name in ("Build_SmelterMk1_C", "Build_Foundation_8x1_01_C"):
        print(f"  {name}: {buildings.get(name, 'MISSING')}")


if __name__ == "__main__":
    main()