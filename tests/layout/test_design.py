import pytest
import torch

from layout.api import evaluate
from layout.design import find_item, make_task, task_name, design
from layout.best_of import EvalTasks
from layout.policy import Policy
from rl.games import ProcGames


def small_task():
    """A small stage 1 task (quick to play)"""
    return make_task("Desc_IronPlate_C", 5)


def test_find_item_takes_names_and_ids():
    assert find_item("Smart Plating") == "Desc_SpaceElevatorPart_1_C"
    assert find_item("Stator") == "Desc_Stator_C"
    assert find_item("Desc_Stator_C") == "Desc_Stator_C"
    with pytest.raises(ValueError):
        find_item("not an item")


def test_make_task_is_repeatable_and_takes_a_site():
    assert make_task("Desc_Stator_C", 5) == make_task("Desc_Stator_C", 5)
    t = make_task("Desc_Stator_C", 5, "64x48")
    assert (t["width"], t["depth"]) == (64, 48)
    assert "64 x 48 m site" in task_name(t)


def test_make_task_refuses_what_the_models_cannot_play():
    with pytest.raises(ValueError):
        make_task("Desc_Stator_C", 5, "200x200")
    with pytest.raises(ValueError):
        make_task("Desc_Computer_C", 120)


@pytest.mark.parametrize("mode, games", [("all", 2 * 4), ("greedy", 2 + 4), ("scout", 2 * 2 + 2)])
def test_design_modes_play_the_right_games_and_keep_a_real_score(mode, games):
    torch.manual_seed(0)
    models = [Policy().eval(), Policy().eval()]
    task = small_task()
    gen = torch.Generator().manual_seed(0)
    with ProcGames(EvalTasks([task]), 2, 1) as g:
        agent, stage, looks, n = design(g, models, mode, gen, k=4, scout=2)
    assert n == games
    assert stage in (1, 2) and len(looks) == 2
    assert agent["score"] >= looks[stage - 1]["score"]
    assert abs(evaluate(task, agent["plan"])["score"] - agent["score"]) < 1e-9