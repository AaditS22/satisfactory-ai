import json
from pathlib import Path

BUILDINGS = {
    "Build_SmelterMk1_C":        "/Game/FactoryGame/Buildable/Factory/SmelterMk1/Build_SmelterMk1.Build_SmelterMk1_C",
    "Build_ConstructorMk1_C":    "/Game/FactoryGame/Buildable/Factory/ConstructorMk1/Build_ConstructorMk1.Build_ConstructorMk1_C",
    "Build_ConveyorBeltMk1_C":   "/Game/FactoryGame/Buildable/Factory/ConveyorBeltMk1/Build_ConveyorBeltMk1.Build_ConveyorBeltMk1_C",
    "Build_Foundation_8x1_01_C": "/Game/FactoryGame/Buildable/Building/Foundation/Build_Foundation_8x1_01.Build_Foundation_8x1_01_C",
}

RECIPES = {
    "Recipe_IngotIron_C": "/Game/FactoryGame/Recipes/Smelter/Recipe_IngotIron.Recipe_IngotIron_C",
    "Recipe_IronPlate_C": "/Game/FactoryGame/Recipes/Constructor/Recipe_IronPlate.Recipe_IronPlate_C",
    "Recipe_Foundation_8x1_01_C": "/Game/FactoryGame/Recipes/Buildings/Foundations/Recipe_Foundation_8x1_01.Recipe_Foundation_8x1_01_C"
}

BUILT_WITH = {
    "Build_Foundation_8x1_01_C": "Recipe_Foundation_8x1_01_C",
}


_GENERATED = Path(__file__).resolve().parent.parent / "data" / "content_paths.json"
if _GENERATED.exists():
    _gen = json.loads(_GENERATED.read_text(encoding="utf-8"))
    BUILDINGS = {**_gen.get("buildings", {}), **BUILDINGS}
    RECIPES = {**_gen.get("recipes", {}), **RECIPES}

def class_path(name):
    try:
        return BUILDINGS[name]
    except KeyError:
        raise KeyError(f"no content path for building {name!r}; add it to executor/paths.py") from None


def recipe_path(name):
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(f"no content path for recipe {name!r}; add it to executor/paths.py") from None


def built_with_path(name):
    recipe = BUILT_WITH.get(name)
    return recipe_path(recipe) if recipe else None