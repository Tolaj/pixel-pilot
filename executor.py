# executor.py
import time
import pyautogui
import subprocess

# safety — prevents pyautogui from going too fast
pyautogui.PAUSE = 0.5
# move mouse to corner to abort
pyautogui.FAILSAFE = True


def click(x: int, y: int) -> str:
    pyautogui.click(x, y)
    return f"clicked ({x}, {y})"


def double_click(x: int, y: int) -> str:
    pyautogui.doubleClick(x, y)
    return f"double clicked ({x}, {y})"


def right_click(x: int, y: int) -> str:
    pyautogui.rightClick(x, y)
    return f"right clicked ({x}, {y})"


def type_text(text: str) -> str:
    pyautogui.typewrite(text, interval=0.05)
    return f"typed: {text}"


def hotkey(*keys: str) -> str:
    if len(keys) == 2:
        pyautogui.keyDown(keys[0])
        pyautogui.press(keys[1])
        pyautogui.keyUp(keys[0])
    else:
        for k in keys:
            pyautogui.keyDown(k)
        for k in reversed(keys):
            pyautogui.keyUp(k)
    return f"hotkey: {'+'.join(keys)}"


def scroll(x: int, y: int, direction: str = "down", clicks: int = 3) -> str:
    amount = -clicks if direction == "down" else clicks
    pyautogui.scroll(amount, x=x, y=y)
    return f"scrolled {direction} at ({x}, {y})"


def press(key: str) -> str:
    pyautogui.press(key)
    return f"pressed: {key}"


def execute_action(action: dict, elements: list) -> str:
    """
    Execute an action from the agent's JSON output.

    action format:
    {
        "action": "click" | "double_click" | "right_click" | "type" | "hotkey" | "scroll" | "press" | "screenshot" | "finished",
        "element_id": 3,       # for click/double_click/right_click/scroll
        "text": "hello",       # for type
        "keys": ["cmd", "space"], # for hotkey
        "key": "enter",        # for press
        "direction": "down",   # for scroll
    }
    """
    action_type = action.get("action")

    # resolve element center if element_id is given
    def get_center():
        eid = action.get("element_id")
        if eid is None:
            raise ValueError("element_id required for this action")
        match = next((e for e in elements if e["id"] == eid), None)
        if match is None:
            raise ValueError(f"element_id {eid} not found")
        return match["center"]

    if action_type == "click":
        cx, cy = get_center()
        return click(cx, cy)

    elif action_type == "double_click":
        cx, cy = get_center()
        return double_click(cx, cy)

    elif action_type == "right_click":
        cx, cy = get_center()
        return right_click(cx, cy)

    elif action_type == "type":
        text = action.get("text", "")
        return type_text(text)

    elif action_type == "hotkey":
        keys = action.get("keys", [])
        return hotkey(*keys)

    elif action_type == "press":
        # if keys list given, treat as hotkey instead
        keys = action.get("keys")
        if keys and len(keys) > 1:
            return hotkey(*keys)
        key = action.get("key") or (keys[0] if keys else "enter")
        return press(key)

    elif action_type == "scroll":
        cx, cy = get_center()
        direction = action.get("direction", "down")
        clicks = action.get("clicks", 3)
        return scroll(cx, cy, direction, clicks)

    elif action_type == "screenshot":
        return "screenshot taken"

    elif action_type == "finished":
        return "finished"

    else:
        raise ValueError(f"Unknown action type: {action_type}")


# quick test
if __name__ == "__main__":
    import time

    time.sleep(2)
    result = hotkey("command", "space")
    print(result)
