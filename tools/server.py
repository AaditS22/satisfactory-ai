import contextlib
import functools
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from tools import registry
from tools import compare
from tools import control

mcp = MCPServer("satisfactory-runs", instructions=(
    "The run registry and training control of the Satisfactory layout RL project. Reading: start with list_runs, "
    "then get_run for one run's summary. Scores are on A-solvable eval tasks; always read the broken count next to "
    "a mean. Verdicts (helps / neutral / worse / inconclusive) follow the project's pre-registered rule. Control: "
    "queue_run / queue_resume add to the queue, start_supervisor works through it, live_status and get_events show "
    "progress, stop_run stops the current training. Greedy eval dips and hovers even in good runs: never stop a "
    "run for that alone. Every new run needs a note, and a prereg text saying what result will mean what."))

def tool(fn):
    """Registers fn as an MCP tool that answers with compact JSON text, or {"error": ...} for a bad request"""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = fn(*args, **kwargs)
        except ValueError as e:
            out = {"error": str(e)}
        return json.dumps(out, separators=(",", ":"))
    return mcp.tool()(run)

def layouts_key(layouts):
    """Checks the 'layouts' input of a tool"""
    if layouts not in ("best", "greedy"):
        raise ValueError("layouts must be 'best' or 'greedy'")
    return layouts

@tool
def list_runs(stage: int = 0) -> str:
    """Lists every run in the registry (one short row each: scores, broken counts, verdict vs the specialist, flags); stage 0 means all stages"""
    rows = [r for r in registry.load_index() if not stage or r["stage"] == stage]
    return {"runs": rows, "specialists": {str(k): v for k, v in registry.specialists().items()}}

@tool
def get_run(run: str, per_task: bool = False) -> str:
    """One run's full summary: greedy and best-of-k scores, comparisons vs A, B and the specialist, rate bands, targets, big-block tasks, penalty parts, broken tasks, worst tasks, training curve and flags; per_task adds one short row per eval task"""
    summary = registry.load_summary(run)
    if summary is None:
        raise ValueError(f"{run} is not in the registry; list_runs shows what is")
    if per_task and "best" in summary:
        summary["per_task"] = [{"seed": t["seed"], "greedy": round(t["greedy"]["score"], 4),
                                "greedy_kind": t["greedy"]["kind"], "best": round(t["best"]["score"], 4),
                                "best_kind": t["best"]["kind"], "blocks": t["best"]["blocks"]}
                               for t in registry.load_tasks(run)["tasks"]]
    return summary

@tool
def compare_runs(a: str, b: str, layouts: str = "best") -> str:
    """Paired comparison a - b on the A-solvable tasks with SE and verdict, by rate band, target and big-block tasks, plus the tasks that differ most; b may be 'A', 'B' or 'specialist'; layouts is 'best' or 'greedy'"""
    return compare.compare_runs(a, b, layouts_key(layouts))

@tool
def compare_set(runs: list[str], baseline: str = "B", layouts: str = "best") -> str:
    """Best real score per task over a set of runs, compared with a baseline ('A', 'B', 'specialist' or a run), and how many tasks each run wins"""
    return compare.compare_set(runs, baseline, layouts_key(layouts))

@tool
def get_task(seed: int, runs: list[str] = [], plans: bool = True) -> str:
    """One eval task in detail: the task, A's and B's layouts and each run's greedy and best layouts with score, kind, block sizes and penalty terms; runs empty means every run of that stage; a plan is a list of [n, rotation, floor, x, y]"""
    return compare.task_view(seed, runs or None, plans)

@tool
def get_config() -> str:
    """The editable settings: tie size, rate bands, flag thresholds and the current specialist of each stage"""
    return {"config": registry.load_config(), "specialists": {str(k): v for k, v in registry.specialists().items()}}

@tool
def live_status() -> str:
    """What is happening right now: whether the supervisor runs, the current run's round, latest numbers, kill rules that are firing, time left, free RAM, GPU, running processes, the queue and the last events"""
    return control.live_status()

@tool
def get_events(since: int = 0, limit: int = 20) -> str:
    """The supervisor's numbered event files (started, milestone, warning, killed, crashed, finished, refused, deadline, queue_empty) with a number above 'since'"""
    return control.get_events(since, limit)

@tool
def list_queue() -> str:
    """The runs waiting in the queue, each with its exact training command and a time estimate"""
    return control.list_queue()

@tool
def queue_run(run: str, stage: int, settings: dict, note: str, prereg: str = "") -> str:
    """Queues a NEW training run; settings uses layout.train's flag names with underscores, e.g. {"rounds": 2000, "workers": 16, "ent_coef": 0.01, "solvable": true, "copies": 8, "groups": 14, "fixed_scale": true, "scale": 0.061034, "seed": 0, "cost_map": true, "init": "runs/other/f_soft_wide.pt"}; note says what the run changes; prereg says what result will mean what; the request is checked and refused with a reason if it breaks a rule"""
    return control.queue_run(run, stage, settings, note, prereg)

@tool
def queue_resume(run: str) -> str:
    """Queues the rest of a run that stopped early, with exactly its own settings read from its run.json (same seed, no init, note or scale)"""
    return control.queue_resume(run)

@tool
def remove_queued(id: int) -> str:
    """Takes one waiting run out of the queue by its id from list_queue"""
    return control.remove_queued(id)

@tool
def start_supervisor(deadline: str = "") -> str:
    """Starts the supervisor in the background so it works through the queue; deadline like "09:00" means: do not start a run that would not finish by then; refused if one is already running or the queue is empty"""
    return control.start_supervisor(deadline)

@tool
def stop_run(reason: str, stop_supervisor: bool = False) -> str:
    """Stops the training that is running now (its training record is kept in the registry as killed, with the reason); stop_supervisor also ends the supervisor, otherwise it goes on with the next queued run"""
    return control.stop_run(reason, stop_supervisor)

@tool
def cleanup_processes(dry_run: bool = True) -> str:
    """Lists leftover game workers whose training program is gone, and stops them when dry_run is false; never touches a training that is still running"""
    return control.cleanup_processes(dry_run)

TOOLS = [list_runs, get_run, compare_runs, compare_set, get_task, get_config, live_status, get_events, list_queue,
         queue_run, queue_resume, remove_queued, start_supervisor, stop_run, cleanup_processes]

def check():
    """Prints what the server would offer and the exact text to register it, to test the setup without Claude"""
    script = str(Path(__file__).resolve())
    print(f"repo      {REPO}")
    print(f"registry  {len(registry.load_index())} run(s): {[r['run'] for r in registry.load_index()]}")
    for t in TOOLS:
        print(f"tool      {t.__name__}")
    entry = {"mcpServers": {"satisfactory-runs": {"command": sys.executable, "args": [script]}}}
    print("\nClaude Desktop, in claude_desktop_config.json:\n" + json.dumps(entry, indent=2))
    print(f'\nClaude Code, run once in this folder:\nclaude mcp add --transport stdio satisfactory-runs -- "{sys.executable}" "{script}"')

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        mcp.run()
