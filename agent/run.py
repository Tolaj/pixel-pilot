"""
GUI Agent — connects to LLM server + GoClick MCP server.

Standalone:
    python -m agent.run "open Safari"

As module:
    from agent.run import run
    run("open Safari")
"""

import argparse
import sys

import config

from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, MCPClient
from mcp import StdioServerParameters

from agent.tools import (
    take_screenshot,
    click_at,
    double_click_at,
    type_text,
    press_key,
    wait_seconds,
)


SYSTEM_PROMPT = """\
You are a GUI automation agent running on macOS. You control the computer by:

1. Taking screenshots to see the current screen state
2. Using goclick_point to find where to click for a given instruction
3. Clicking, typing, and pressing keys to interact

Workflow for each step:
1. screenshot_path = take_screenshot()
2. result = goclick_point(image_path=screenshot_path, instruction="click on X")
3. Parse the JSON result to get x, y coordinates
4. click_at(x=int(x), y=int(y))
5. wait_seconds(seconds=1.0)
6. Take another screenshot to verify

If an action doesn't work (screen didn't change), try a different instruction
for goclick_point or try a different approach.

Important:
- Always take a screenshot FIRST before deciding what to do
- goclick_point returns a JSON string — parse it with json.loads()
- Keep actions simple: one click or one type per step
- After typing a URL, press_key("return") to navigate
"""


def run(task, port=None, max_steps=None, spawn_mcp=True):
    """
    Run the agent.

    Args:
        task: What to do.
        port: LLM server port.
        max_steps: Max agent steps.
        spawn_mcp: If True, spawns GoClick as subprocess (option 1).
                   If False, connects to already-running MCP server (option 2).
    """
    port = port or config.LLM_PORT
    max_steps = max_steps or config.MAX_STEPS

    print(f"Task: {task}")

    native_tools = [
        take_screenshot, click_at, double_click_at,
        type_text, press_key, wait_seconds,
    ]

    if spawn_mcp:
        mcp_config = [StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
        )]
    else:
        from smolagents import MCPClient as _  # noqa
        mcp_config = [{"url": config.MCP_URL}]

    with MCPClient(mcp_config) as mcp_tools:
        print(f"MCP tools: {[t.name for t in mcp_tools]}")

        model_id = (config.LLM_MODEL_MLX if config.LLM_BACKEND == "mlx"
                    else config.LLM_MODEL_GGUF)
        model = OpenAIServerModel(
            model_id=model_id,
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="not-needed",
        )

        agent = ToolCallingAgent(
            tools=native_tools + mcp_tools,
            model=model,
            max_steps=max_steps,
        )

        result = agent.run(f"{SYSTEM_PROMPT}\n\nTask: {task}")
        print(f"\nResult: {result}")
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
