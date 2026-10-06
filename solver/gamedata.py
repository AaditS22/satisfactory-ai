import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "game_data.json"

with open(DATA_PATH) as f:
    _data = json.load(f)

items = _data["items"]
recipes = _data["recipes"]
recipe_by_id = {r["id"]: r for r in recipes}