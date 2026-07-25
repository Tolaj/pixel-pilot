"""
Native tools for the GUI agent — screenshot, click, type.

These run on macOS using PyObjC / AppleScript.
"""

import json
import os
import subprocess
import tempfile
import time

from smolagents import tool


@tool
def take_screenshot() -> str:
    """
    Take a screenshot of the current macOS screen and save it to a temp file.

    Returns:
        Path to the saved screenshot PNG file.
    """
    path = os.path.join(tempfile.gettempdir(), f"pixel_pilot_screen_{int(time.time())}.png")
    result = subprocess.run(["screencapture", "-x", path], capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "could not create image" in stderr:
            return "ERROR: Screen recording permission denied. Grant it in System Settings > Privacy & Security > Screen Recording for your terminal app, then restart the terminal."
        return f"ERROR: screencapture failed: {stderr}"
    if not os.path.exists(path):
        return "ERROR: Screenshot file was not created."
    return path


@tool
def click_at(x: int, y: int) -> str:
    """
    Click the mouse at screen coordinates (x, y).

    Args:
        x: Horizontal pixel coordinate from left edge.
        y: Vertical pixel coordinate from top edge.

    Returns:
        Confirmation message.
    """
    script = f'''
    tell application "System Events"
        click at {{{x}, {y}}}
    end tell
    '''
    # Use cliclick for reliable clicking (brew install cliclick)
    try:
        subprocess.run(["cliclick", f"c:{x},{y}"], check=True, timeout=5)
        return f"Clicked at ({x}, {y})"
    except FileNotFoundError:
        # Fallback to AppleScript
        try:
            subprocess.run(["osascript", "-e", script], check=True, timeout=5)
            return f"Clicked at ({x}, {y}) via AppleScript"
        except Exception as e:
            return f"Click failed: {e}"


@tool
def double_click_at(x: int, y: int) -> str:
    """
    Double-click the mouse at screen coordinates (x, y).

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.

    Returns:
        Confirmation message.
    """
    try:
        subprocess.run(["cliclick", f"dc:{x},{y}"], check=True, timeout=5)
        return f"Double-clicked at ({x}, {y})"
    except FileNotFoundError:
        return "cliclick not installed. Run: brew install cliclick"


@tool
def type_text(text: str) -> str:
    """
    Type text using the keyboard at the current cursor position.

    Args:
        text: The text string to type.

    Returns:
        Confirmation message.
    """
    try:
        subprocess.run(["cliclick", f"t:{text}"], check=True, timeout=10)
        return f"Typed: {text}"
    except FileNotFoundError:
        script = f'tell application "System Events" to keystroke "{text}"'
        try:
            subprocess.run(["osascript", "-e", script], check=True, timeout=10)
            return f"Typed: {text}"
        except Exception as e:
            return f"Type failed: {e}"


@tool
def press_key(key: str) -> str:
    """
    Press a special key like 'return', 'tab', 'escape', 'space', 'delete'.

    Args:
        key: Key name — return, tab, escape, space, delete, up, down, left, right.

    Returns:
        Confirmation message.
    """
    key_map = {
        "return": "kp:return",
        "enter": "kp:return",
        "tab": "kp:tab",
        "escape": "kp:escape",
        "esc": "kp:escape",
        "space": "kp:space",
        "delete": "kp:delete",
        "backspace": "kp:delete",
        "up": "kp:arrow-up",
        "down": "kp:arrow-down",
        "left": "kp:arrow-left",
        "right": "kp:arrow-right",
    }

    cmd = key_map.get(key.lower())
    if not cmd:
        return f"Unknown key: {key}. Supported: {', '.join(key_map.keys())}"

    try:
        subprocess.run(["cliclick", cmd], check=True, timeout=5)
        return f"Pressed: {key}"
    except FileNotFoundError:
        return "cliclick not installed. Run: brew install cliclick"


@tool
def wait_seconds(seconds: float) -> str:
    """
    Wait for a specified number of seconds before the next action.

    Args:
        seconds: How many seconds to wait (0.5 to 10).

    Returns:
        Confirmation message.
    """
    seconds = max(0.5, min(10, seconds))
    time.sleep(seconds)
    return f"Waited {seconds}s"
