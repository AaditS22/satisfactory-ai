import math

from solver.gamedata import recipe_by_id
from solver.graph import BELT_CAPACITY

def depths(graph):
    """Longest number of edges from a recipe to output"""
    nexts = {}

    for edge in graph["edges"]:
        nexts.setdefault(edge["from"], []).append(edge["to"])

    depths_dict = {}

    for node in graph["nodes"]:
        fill_depth(depths_dict, node, nexts)

    return depths_dict

def fill_depth(depths_dict, node, nexts):
    """Helper method to fill in depths dict while finding depths recursively"""

    # Base case, last node
    if "OUTPUT" in nexts[node]:
        depths_dict[node] = 0
        return 0

    # Already found depth for this in previous iteration
    if node in depths_dict:
        return depths_dict[node]

    max_depth = -1
    for neighbour in nexts[node]:
        next_depth = fill_depth(depths_dict, neighbour, nexts)
        max_depth = max(max_depth, next_depth)

    depths_dict[node] = max_depth + 1
    return max_depth + 1    

def make_queue(graph):
    """Makes a queue of recipes in order of placement"""

    depths_list = depths(graph)
    sorted_nodes = sorted(graph["nodes"], key=lambda r:((-depths_list[r], r)))

    queue = []
    for node in sorted_nodes:
        queue.append({
            "recipe": node,
            "machine": graph["nodes"][node]["machine"],
            "clock": graph["nodes"][node]["clocks"][0],
            "remaining": len(graph["nodes"][node]["clocks"])
        })

    return queue

def max_block(entry, max_tier=6):
    """The max num of machines of a recipe that can go in one block without overloading belt"""

    recipe = recipe_by_id[entry["recipe"]]
    max_rate = max(list(recipe["inputs"].values()) + list(recipe["outputs"].values()))
    capacity = BELT_CAPACITY[max_tier - 1]

    x = capacity / (max_rate * entry["clock"])

    return math.floor(x + 1e-9)

def take(queue, n, max_tier=6):
    """Place a block of n machines from the head of the queue. Raises ValueError if n is not allowed."""

    head = queue[0]

    limit = min(head["remaining"], max_block(head, max_tier))
    if n < 1 or n > limit:
        raise ValueError(f"n is not within bounds: 1 to {limit}")

    head["remaining"] -= n
    if head["remaining"] == 0:
        queue.pop(0)

    return (head["recipe"], n, head["clock"])