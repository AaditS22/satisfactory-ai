import json

from layout.env import LayoutEnv
from layout.wire import route_all
from layout.score import score_episode

def plan_of(env):
    return [[int(b["block"]["n"]), int(b["block"]["rot"]), int(b["floor"]), int(b["x"]), int(b["y"])]
        for b in env.world.blocks]

def play(task, plan):
    env = LayoutEnv()
    env.reset(task)
    for move in plan:
        env.step(*move)
    return env

def evaluate(task, plan):
    env = play(task, plan)
    if not env.done:
        raise ValueError("plan does not finish the episode")
    routes = None if env.failed else route_all(env)
    s, terms = score_episode(env, routes)
    return {"env": env, "routes": routes, "score": s, "terms": terms}

def save(path, task, plan):
    t = dict(task)
    if t.get("alternates") is not None:
        t["alternates"] = sorted(t["alternates"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"task": t, "plan": plan}, fh, indent=1)

def load(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    task = data["task"]
    if task.get("alternates") is not None:
        task["alternates"] = frozenset(task["alternates"])
    return task, data["plan"]