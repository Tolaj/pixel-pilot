# agent.py
import subprocess
import time
import base64
import json
import io
import atexit
import requests
from PIL import Image

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "./models/Qwen3-VL-2B-Instruct-GGUF/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
SERVER_URL = "http://localhost:8080"
_server_process = None


def start_server():
    global _server_process
    print("Starting Qwen3-VL server...")
    _server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "llama_cpp.server",
            "--model",
            MODEL_PATH,
            "--clip_model_path",
            MMPROJ_PATH,
            "--n_gpu_layers",
            "-1",
            "--n_ctx",
            "8192",
            "--port",
            "8080",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)

    # wait until server is ready
    for _ in range(30):
        try:
            requests.get(f"{SERVER_URL}/health", timeout=2)
            print("Server ready.")
            return
        except:
            time.sleep(2)
    raise RuntimeError("Server failed to start")


def stop_server():
    global _server_process
    if _server_process:
        print("Stopping server...")
        _server_process.terminate()
        _server_process = None


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


KEY_MAP = {
    "cmd": "command",
    "ctrl": "control",
    "alt": "option",
    "return": "enter",
}


def parse_action(content: str) -> dict:
    try:
        content = content.strip()
        # find first { and its matching }
        first_brace = content.index("{")
        # find matching closing brace by counting depth
        depth = 0
        end = first_brace
        for i, ch in enumerate(content[first_brace:], start=first_brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        content = content[first_brace : end + 1]
        action = json.loads(content)

        # normalize keys
        keys = action.get("keys")
        if isinstance(keys, str):
            action["keys"] = [k.strip() for k in keys.replace("+", " ").split()]
        if action.get("keys"):
            KEY_MAP = {
                "cmd": "command",
                "ctrl": "control",
                "alt": "option",
                "return": "enter",
            }
            action["keys"] = [KEY_MAP.get(k.lower(), k.lower()) for k in action["keys"]]

        # normalize element_id
        eid = action.get("element_id")
        if isinstance(eid, str):
            action["element_id"] = int(eid) if eid.isdigit() else None

        return action

    except Exception as e:
        print(f"Failed to parse response: {content}\nError: {e}")
        return {
            "action": "screenshot",
            "thought": "parse error retrying",
            "finished": False,
        }


SYSTEM_PROMPT = """You are a GUI agent on a Mac. Given a task, screenshot, and UI elements list, respond with the next action as JSON only, no extra text:
{"thought": "...", "action": "click|double_click|right_click|type|hotkey|press|scroll|screenshot|finished", "element_id": null, "text": null, "keys": ["cmd", "space"], "key": null, "direction": null, "finished": false}

Rules:
- NEVER use action "press" for combinations like cmd+space, use "hotkey" instead
- "press" is only for single keys like enter, escape, tab
- "hotkey" is for combinations like ["cmd", "space"], ["cmd", "c"]
- keys must always be a list e.g. ["cmd", "space"] never a string
- element_id must be an integer or null
- after typing in a search bar always press enter to confirm
- only return finished when you can see the result on screen, not just because you typed something

{"thought": "...", "action": "click|type|hotkey|press|scroll|screenshot|finished", "element_id": null, "text": null, "keys": null, "key": null, "direction": null, "finished": false}"""


def get_next_action(
    task: str,
    screenshot: Image.Image,
    elements: list,
    history: list,
) -> dict:
    img_b64 = image_to_base64(screenshot)
    elements = elements[:20]
    elements_text = "\n".join(
        f"[{e['id']}] label='{e['label']}' center={e['center']}" for e in elements
    )
    history_text = ""
    if history:
        history_text = "\nPrevious actions:\n" + "\n".join(
            f"  step {i+1}: {h}" for i, h in enumerate(history[-5:])
        )

    user_message = f"""Task: {task}

Detected UI elements:
{elements_text}
{history_text}

What is the next action?"""

    payload = {
        "model": "qwen3-vl",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    print("Status:", response.status_code)
    print("Response:", response.text)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"].strip()

    # parse JSON from response
    return parse_action(content)


if __name__ == "__main__":
    start_server()
    from perception import parse_screen, elements_to_text

    result = parse_screen()
    action = get_next_action(
        task="Open Spotlight search",
        screenshot=result["screenshot"],
        elements=result["elements"],
        history=[],
    )
    print("Agent action:", json.dumps(action, indent=2))
    stop_server()
