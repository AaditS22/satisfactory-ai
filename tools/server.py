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

mcp = MCPServer("satisfactory-runs", instructions=(
    "Read-only access to the run registry of the Satisfactory layout RL project. Start with list_runs, then "
    "get_run for one run's summary. Scores are on A-solvable eval tasks; always read the broken count next to a "
    "mean. Verdicts (helps / neutral / worse / inconclusive) follow the project's pre-registered rule."))

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
    if per_task:
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

TOOLS = [list_runs, get_run, compare_runs, compare_set, get_task, get_config]

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
