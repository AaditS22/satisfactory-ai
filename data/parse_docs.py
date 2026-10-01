import json
import re

AMOUNT_RE = re.compile(r"\.(\w+_C)\W*,\s*Amount=(\d+)")
PRODUCER_RE = re.compile(r"\.(Build_\w+_C)")

def native_name(group):
    return group["NativeClass"].split(".")[-1].strip("'\"")

def parse_amounts(s):
    return {item: int(n) for item, n in AMOUNT_RE.findall(s)}

def per_minute(amount_str, duration):
    result = {}
    for item, n in parse_amounts(amount_str).items():
        form = items.get(item, {}).get("form")
        scale = 1000 if form in ("RF_LIQUID", "RF_GAS") else 1
        result[item] = n / scale * 60 / duration
    return result

if __name__ == "__main__":
    with open("data/raw/en-US.json", encoding="utf-16") as f:
        docs = json.load(f)

    items = {}
    for group in docs:
        for c in group["Classes"]:
            if "mForm" in c:
                items[c["ClassName"]] = {"name": c.get("mDisplayName", ""), "form": c["mForm"]}

    recipe_group = next(g for g in docs if native_name(g) == "FGRecipe")
    recipes = []
    for c in recipe_group["Classes"]:
        machines = [m for m in PRODUCER_RE.findall(c["mProducedIn"]) if m != "Build_AutomatedWorkBench_C"]
        if not machines:
            continue
        duration = float(c["mManufactoringDuration"])
        recipes.append({
            "id": c["ClassName"],
            "name": c["mDisplayName"],
            "alternate": c["ClassName"].startswith("Recipe_Alternate"),
            "machine": machines[0],
            "duration_s": duration,
            "inputs": per_minute(c["mIngredients"], duration),
            "outputs": per_minute(c["mProduct"], duration),
        })

    with open("data/game_data.json", "w") as f:
        json.dump({"items": items, "recipes": recipes}, f, indent=2)

    print("Parsed successfully")             