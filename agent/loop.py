"""
Custom agent loop — sends plain string messages compatible with llama-cpp-python.

smolagents sends content as [{"type":"text","text":"..."}] which llama.cpp rejects.
This loop uses urllib with plain string content, and parses tool calls with regex.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request

import config


SYSTEM_PROMPT = """\
You are a macOS GUI automation agent. You have these tools:

TOOLS:
- take_screenshot() -> returns file path of screenshot
- goclick_point(image_path, instruction) -> returns JSON with x, y coordinates
- click_at(x, y) -> clicks at screen coordinates
- double_click_at(x, y) -> double clicks
- type_text(text) -> types text at cursor
- press_key(key) -> presses key (return, tab, escape, space, delete, up, down, left, right)
- wait_seconds(seconds) -> waits
- goclick_health() -> checks if GoClick model is ready

RULES:
1. Call ONE tool per response
2. Respond with EXACTLY this format:

THOUGHT: what you're doing and why
TOOL: tool_name
ARGS: {"param": "value"}

3. Always take_screenshot FIRST to see the screen
4. To find where to click: call goclick_point with the screenshot path and a SPECIFIC instruction
   - Be specific about WHAT and WHERE: "click on Chrome icon in the dock at the bottom" not just "click on Chrome"
   - Include location hints: "in the dock", "in the menu bar", "in the center of the window"
5. Parse goclick_point result to get x,y then call click_at
6. After EVERY click, call wait_seconds then take_screenshot to VERIFY the result
7. Only call "done" AFTER you have verified with a screenshot that the task succeeded
8. The macOS dock is at the BOTTOM of the screen. App icons are in the dock.

WORKFLOW for every click action:
  take_screenshot → goclick_point → click_at → wait_seconds → take_screenshot (verify) → done or next action

Example:
THOUGHT: I need to see the screen first
TOOL: take_screenshot
ARGS: {}

Example:
THOUGHT: I need to find Chrome icon in the dock at the bottom of the screen
TOOL: goclick_point
ARGS: {"image_path": "/tmp/screenshot.png", "instruction": "click on Chrome icon in the dock at the bottom of the screen"}

Example:
THOUGHT: GoClick found Chrome at x=500, y=900. Clicking there.
TOOL: click_at
ARGS: {"x": 500, "y": 900}

Example:
THOUGHT: I clicked, now I need to wait and verify.
TOOL: wait_seconds
ARGS: {"seconds": 1.0}

When the task is verified DONE (after a verification screenshot), respond with:
THOUGHT: Task complete — verified Chrome is open.
TOOL: done
ARGS: {}
"""


def _parse_tool_call(text):
    tool_match = re.search(r'TOOL:\s*(\w+)', text)
    if not tool_match:
        for known in ['take_screenshot', 'goclick_point', 'goclick_point_base64',
                       'goclick_health', 'click_at', 'double_click_at',
                       'type_text', 'press_key', 'wait_seconds', 'done']:
            if known in text:
                tool_match = re.search(rf'\b({re.escape(known)})\b', text)
                if tool_match:
                    break

    if not tool_match:
        return None, None

    tool_name = tool_match.group(1)

    args_match = re.search(r'ARGS:\s*(\{.*?\})', text, re.DOTALL)
    if not args_match:
        args_match = re.search(r'\{[^{}]*\}', text)

    args = {}
    if args_match:
        try:
            raw = args_match.group(1) if args_match.lastindex else args_match.group(0)
            args = json.loads(raw)
        except json.JSONDecodeError:
            raw = args_match.group(1) if args_match.lastindex else args_match.group(0)
            for kv in re.finditer(r'"(\w+)"\s*:\s*(".*?"|[\d.]+|\btrue\b|\bfalse\b)', raw):
                key = kv.group(1)
                val = kv.group(2)
                if val.startswith('"'):
                    args[key] = val.strip('"')
                elif '.' in val:
                    args[key] = float(val)
                elif val in ('true', 'false'):
                    args[key] = val == 'true'
                else:
                    args[key] = int(val)

    return tool_name, args


def _extract_thought(text):
    m = re.search(r'THOUGHT:\s*(.+?)(?:\n|TOOL:|$)', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _chat(messages, port):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    model_id = (config.LLM_MODEL_MLX if config.LLM_BACKEND == "mlx"
                else config.LLM_MODEL_GGUF)
    body = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


def run(task, tools, max_steps=None, port=None):
    port = port or config.LLM_PORT
    max_steps = max_steps or config.MAX_STEPS

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    print(f"\n{'='*50}")
    print(f"  Task: {task}")
    print(f"  Max steps: {max_steps}")
    print(f"{'='*50}\n")

    for step in range(1, max_steps + 1):
        print(f"--- Step {step}/{max_steps} ---")
        t0 = time.time()

        try:
            response = _chat(messages, port)
        except Exception as e:
            print(f"  LLM error: {e}")
            break

        elapsed = time.time() - t0
        thought = _extract_thought(response)
        tool_name, args = _parse_tool_call(response)

        if thought:
            print(f"  Thought: {thought}")

        if not tool_name:
            print(f"  Could not parse tool call from: {response[:200]}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                "I couldn't understand your response. Use this exact format:\nTHOUGHT: ...\nTOOL: tool_name\nARGS: {}"
            })
            continue

        if tool_name == "done":
            print(f"\n  Task complete! ({elapsed:.1f}s)")
            return thought

        if tool_name not in tools:
            print(f"  Unknown tool: {tool_name}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"Unknown tool '{tool_name}'. Available: {', '.join(tools.keys())}"
            })
            continue

        print(f"  Tool: {tool_name}({args})")

        try:
            result = tools[tool_name](**args)
            result_str = str(result)
        except Exception as e:
            result_str = f"Error: {e}"

        print(f"  Result: {result_str[:200]}")
        print(f"  ({elapsed:.1f}s)")

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool result: {result_str}"})

    print("\n  Max steps reached.")
    return None
