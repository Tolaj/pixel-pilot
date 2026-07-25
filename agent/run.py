"""
GUI Agent — custom loop with GoClick MCP server.

Standalone:
    python -m agent.run "open Safari"

As module:
    from agent.run import run
    run("open Safari")
"""

import argparse
import sys

import config

from agent.loop import run as loop_run
from agent.tools import (
    take_screenshot,
    click_at,
    double_click_at,
    type_text,
    press_key,
    wait_seconds,
)


def _make_tools(spawn_mcp):
    """Build a tools dict for the custom loop."""
    tools = {
        "take_screenshot": lambda **kw: take_screenshot(),
        "click_at": lambda **kw: click_at(x=int(kw["x"]), y=int(kw["y"])),
        "double_click_at": lambda **kw: double_click_at(x=int(kw["x"]), y=int(kw["y"])),
        "type_text": lambda **kw: type_text(text=kw["text"]),
        "press_key": lambda **kw: press_key(key=kw["key"]),
        "wait_seconds": lambda **kw: wait_seconds(seconds=float(kw["seconds"])),
    }

    if spawn_mcp:
        from mcp_server.server import _ensure_model, goclick_point, goclick_health

        def _goclick_point(**kw):
            _ensure_model()
            return goclick_point(image_path=kw["image_path"], instruction=kw["instruction"])

        def _goclick_health(**kw):
            _ensure_model()
            return goclick_health()

        tools["goclick_point"] = _goclick_point
        tools["goclick_health"] = _goclick_health
    else:
        from smolagents import MCPClient
        mcp_client = MCPClient([{"url": config.MCP_URL}])
        mcp_tools_list = mcp_client.__enter__()
        mcp_map = {t.name: t for t in mcp_tools_list}
        print(f"MCP tools: {list(mcp_map.keys())}")

        def _goclick_point(**kw):
            return mcp_map["goclick_point"](image_path=kw["image_path"], instruction=kw["instruction"])

        def _goclick_health(**kw):
            return mcp_map["goclick_health"]()

        tools["goclick_point"] = _goclick_point
        tools["goclick_health"] = _goclick_health
        tools["_mcp_client"] = mcp_client

    return tools


def run(task, port=None, max_steps=None, spawn_mcp=True):
    port = port or config.LLM_PORT
    max_steps = max_steps or config.MAX_STEPS

    print(f"Task: {task}")
    tools = _make_tools(spawn_mcp)

    loop_tools = {k: v for k, v in tools.items() if not k.startswith("_")}

    result = loop_run(task, loop_tools, max_steps=max_steps, port=port)

    mcp_client = tools.get("_mcp_client")
    if mcp_client:
        mcp_client.__exit__(None, None, None)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--port", type=int, default=config.LLM_PORT)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--no-spawn-mcp", action="store_true",
                        help="Connect to existing MCP server instead of spawning one")
    args = parser.parse_args()
    run(args.task, args.port, args.max_steps, spawn_mcp=not args.no_spawn_mcp)
