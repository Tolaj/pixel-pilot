# PromptPack Output

**Root:** `/Users/swapnil/Documents/Projects/pixel-pilot`
**Generated:** 2026-05-05T00:30:21.132Z

---

## 1) Folder Structure

```txt
.
├─ agent.py
├─ download_models.py
├─ executor.py
├─ loop.py
├─ main.py
├─ models.py
├─ output.png
├─ perception.py
├─ requirements.txt
├─ test_omniparser.py
├─ test_screenshot.png
└─ test.py
```

<!-- PAGE BREAK: FILE CONTENTS BELOW -->

## 2) File Contents


### agent.py

```python
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

```

### download_models.py

```python
from huggingface_hub import hf_hub_download, snapshot_download
import os

# Download full Florence-2 base (processor + config + weights)
print("Downloading Florence-2 base...")
snapshot_download(
    repo_id="microsoft/Florence-2-base-ft",
    local_dir="./models/omniparser/icon_caption_florence",
)

# Override weights with OmniParser v2 fine-tuned model
print("Downloading OmniParser v2 fine-tuned caption weights...")
hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_caption/model.safetensors",
    local_dir="./models/omniparser",
)

# Replace the weights
import shutil

src = "./models/omniparser/icon_caption/model.safetensors"
dst = "./models/omniparser/icon_caption_florence/model.safetensors"
shutil.copy(src, dst)
print(f"Replaced weights: {dst}")

```

### executor.py

```python
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

```

### loop.py

```python
# loop.py
import time
from perception import parse_screen
from agent import start_server, stop_server, get_next_action
from executor import execute_action

MAX_STEPS = 20
SCREENSHOT_DELAY = 1.5  # seconds to wait after action before next screenshot


def run(task: str):
    print(f"\nTask: {task}")
    print("=" * 50)

    start_server()

    history = []
    step = 0

    try:
        while step < MAX_STEPS:
            step += 1
            print(f"\n--- Step {step}/{MAX_STEPS} ---")

            # perceive
            print("Taking screenshot and parsing screen...")
            result = parse_screen()
            elements = result["elements"]
            screenshot = result["screenshot"]
            print(f"Found {len(elements)} elements in {result['parse_time']}s")

            # think
            print("Asking agent for next action...")
            action = get_next_action(
                task=task,
                screenshot=screenshot,
                elements=elements,
                history=history,
            )
            print(f"Thought: {action.get('thought', '')}")
            print(
                f"Action: {action.get('action')} | element_id={action.get('element_id')} | text={action.get('text')} | keys={action.get('keys')} | key={action.get('key')}"
            )

            # check if done
            if action.get("action") == "finished" or action.get("finished"):
                print("\nTask completed!")
                break

            # execute
            try:
                result_msg = execute_action(action, elements)
                print(f"Executed: {result_msg}")
                history.append(f"{action.get('action')}: {result_msg}")
            except Exception as e:
                print(f"Execution error: {e}")
                history.append(f"error: {e}")

            # wait before next screenshot
            time.sleep(SCREENSHOT_DELAY)

        else:
            print(f"\nReached max steps ({MAX_STEPS}), stopping.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        stop_server()


if __name__ == "__main__":
    run("Open Spotlight search and search for Calculator")

```

### main.py

```python
# main.py
import sys
from loop import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        task = input("What do you want me to do? ")
    else:
        task = " ".join(sys.argv[1:])

    run(task)

```

### models.py

```python
import os
from huggingface_hub import hf_hub_download


def download_if_missing(repo_id, files, local_dir):
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            print(f"Already exists, skipping: {local_path}")
            continue
        print(f"Downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"Done: {local_path}")


# Qwen3-VL-2B-Instruct
download_if_missing(
    repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
    local_dir="./models/Qwen3-VL-2B-Instruct-GGUF",
    files=[
        "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
    ],
)

# OmniParser
OMNI_DIR = "./models/omniparser"
download_if_missing(
    repo_id="microsoft/OmniParser-v2.0",
    local_dir=OMNI_DIR,
    files=[
        "icon_detect/train_args.yaml",
        "icon_detect/model.pt",
        "icon_detect/model.yaml",
        "icon_caption/config.json",
        "icon_caption/generation_config.json",
        "icon_caption/model.safetensors",
    ],
)

# Rename icon_caption → icon_caption_florence if not done yet
src = os.path.join(OMNI_DIR, "icon_caption")
dst = os.path.join(OMNI_DIR, "icon_caption_florence")
if os.path.exists(src) and not os.path.exists(dst):
    os.rename(src, dst)
    print("Renamed icon_caption → icon_caption_florence")
elif os.path.exists(dst):
    print("icon_caption_florence already exists, skipping rename")

```

### output.png

(Skipped: binary or unreadable file)


### perception.py

```python
# perception.py
import os
import time
import torch
from PIL import Image
import pyautogui
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoProcessor

OMNI_DIR = "./models/omniparser"
MIN_CONF = 0.5
ICON_DETECT = os.path.join(OMNI_DIR, "icon_detect/model.pt")
ICON_CAPTION = os.path.join(OMNI_DIR, "icon_caption_florence")
FLORENCE_SAFE = os.path.join(ICON_CAPTION, "model.safetensors")

# FIX 4 (Mac): stop flash_attn from being imported — must be set before torch/transformers load
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

_yolo = None
_processor = None
_florence = None

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FLORENCE_DTYPE = torch.float16


def _ensure_safetensors():
    """FIX 2: convert pytorch_model.bin → model.safetensors on first run."""
    if os.path.exists(FLORENCE_SAFE):
        return
    print("Converting Florence-2 to safetensors (one-time, takes ~30s)...")
    tmp = AutoModelForCausalLM.from_pretrained(
        ICON_CAPTION,
        torch_dtype=FLORENCE_DTYPE,
        trust_remote_code=True,
    )
    tmp.save_pretrained(ICON_CAPTION)  # writes model.safetensors in-place
    del tmp
    torch.mps.empty_cache() if DEVICE == "mps" else None
    print("Conversion done — future loads will be faster.")


def _load_models():
    global _yolo, _processor, _florence

    if _yolo is None:
        print("Loading YOLO (cpu)...")
        # FIX 4 (Mac): YOLO on MPS is 54× slower than CUDA due to MPS kernel overhead;
        # CPU is faster on Apple Silicon for convolutional inference
        _yolo = YOLO(ICON_DETECT)
        _yolo.to("cpu")

    if _florence is None:
        _ensure_safetensors()
        print(f"Loading Florence-2 ({DEVICE})...")
        _processor = AutoProcessor.from_pretrained(ICON_CAPTION, trust_remote_code=True)
        _florence = AutoModelForCausalLM.from_pretrained(
            ICON_CAPTION,
            torch_dtype=FLORENCE_DTYPE,
            trust_remote_code=True,
            # FIX 4 (Mac): attn_implementation="eager" disables flash_attn which
            # hangs the MPS backend and causes multi-second stalls on Mac
            attn_implementation="eager",
        ).to(DEVICE)
        _florence.eval()


def _caption_batch(crops: list[Image.Image], batch_size: int = 32) -> list[str]:
    """
    FIX (throughput): caption all crops in batches rather than one at a time.
    Cuts Metal/CUDA dispatch overhead from N calls → ceil(N/batch_size) calls.
    batch_size=32 is safe for 8 GB unified memory; lower if you see OOM errors.
    """
    results = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        inputs = _processor(
            text=["<CAPTION>"] * len(batch),
            images=batch,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE, FLORENCE_DTYPE)

        with torch.inference_mode():
            generated_ids = _florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,  # UI labels are short; was 64 — saves decode time
                num_beams=1,  # greedy decode; was 3 — 3× faster, negligible quality loss
            )

        decoded = _processor.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend([s.strip() or "icon" for s in decoded])

    return results


def take_screenshot() -> Image.Image:
    return pyautogui.screenshot()


def parse_screen(image: Image.Image = None) -> dict:
    _load_models()

    if image is None:
        image = take_screenshot()

    image_rgb = image.convert("RGB")
    start = time.time()

    results = _yolo(image_rgb, verbose=False)
    boxes = results[0].boxes

    # filter by confidence and collect crops in one pass
    kept_boxes = []
    crops = []
    for box in boxes:
        conf = box.conf[0].item()
        if conf < MIN_CONF:
            continue
        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
        kept_boxes.append((x1, y1, x2, y2, round(conf, 2)))
        crops.append(image_rgb.crop((x1, y1, x2, y2)))

    # caption everything in batches
    captions = _caption_batch(crops) if crops else []

    elements = []
    for (x1, y1, x2, y2, conf), label in zip(kept_boxes, captions):
        cx, cy = (x1 + x2) // 4, (y1 + y2) // 4
        elements.append(
            {
                "id": len(elements),
                "label": label,
                "center": [cx, cy],
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
            }
        )

    elapsed = time.time() - start
    return {
        "elements": elements,
        "count": len(elements),
        "screenshot": image,
        "parse_time": round(elapsed, 2),
    }


def elements_to_text(elements: list) -> str:
    lines = [
        f"[{el['id']}] label='{el['label']}' center={el['center']}" for el in elements
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = parse_screen()
    print(f"Parsed {result['count']} elements in {result['parse_time']}s\n")
    print(elements_to_text(result["elements"]))

```

### requirements.txt

```text
annotated-doc==0.0.4
anyio==4.13.0
certifi==2026.4.22
click==8.3.3
diskcache==5.6.3
filelock==3.29.0
fsspec==2026.4.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.13.0
idna==3.13
jinja2==3.1.6

# llama-cpp-python==0.3.21

# CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache-dir

markdown-it-py==4.0.0
markupsafe==3.0.3
mdurl==0.1.2
numpy==2.4.4
packaging==26.2
pygments==2.20.0
pyyaml==6.0.3
rich==15.0.0
shellingham==1.5.4
tqdm==4.67.3
typer==0.25.1
typing-extensions==4.15.0

```

### test_omniparser.py

```python
import os
import torch
from PIL import Image
import pyautogui
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import ImageDraw, ImageFont

# paths
OMNI_DIR = "./models/omniparser"
ICON_DETECT = os.path.join(OMNI_DIR, "icon_detect/model.pt")
ICON_CAPTION = os.path.join(OMNI_DIR, "icon_caption_florence")


# load models
print("Loading YOLO...")
yolo = YOLO(ICON_DETECT)

print("Loading Florence-2...")
processor = AutoProcessor.from_pretrained(ICON_CAPTION, trust_remote_code=True)
florence = AutoModelForCausalLM.from_pretrained(
    ICON_CAPTION,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("mps")


def caption_element(image_crop: Image.Image) -> str:
    """Run Florence-2 on a cropped UI element to get its description."""
    inputs = processor(text="<CAPTION>", images=image_crop, return_tensors="pt").to(
        "mps", torch.float16
    )

    with torch.inference_mode():
        generated_ids = florence.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=3,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result


def run_omniparser(image: Image.Image):
    """Full OmniParser pipeline: detect + caption every UI element."""
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size

    # step 1: YOLO detection
    results = yolo(image_rgb)
    boxes = results[0].boxes

    print(f"\nDetected {len(boxes)} elements\n")
    print(f"{'ID':<5} {'Conf':<6} {'BBox':<35} {'Caption'}")
    print("-" * 80)

    elements = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
        conf = box.conf[0].item()

        # step 2: crop and caption with Florence-2
        crop = image_rgb.crop((x1, y1, x2, y2))
        caption = caption_element(crop)

        elements.append(
            {
                "id": i,
                "bbox": [x1, y1, x2, y2],
                "conf": round(conf, 2),
                "caption": caption,
            }
        )

        print(f"{i:<5} {conf:<6.2f} {str([x1,y1,x2,y2]):<35} {caption}")

    return elements


# take screenshot and run
print("Taking screenshot...")
screenshot = pyautogui.screenshot()
screenshot.save("test_screenshot.png")

elements = run_omniparser(screenshot)
print(f"\nTotal elements parsed: {len(elements)}")


def save_annotated(image: Image.Image, elements: list, out_path="output.png"):
    draw = ImageDraw.Draw(image)

    for el in elements:
        x1, y1, x2, y2 = el["bbox"]
        # draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # draw id + caption
        label = f"[{el['id']}] {el['caption'][:30]}"
        draw.text((x1, y1 - 12), label, fill="red")

    image.save(out_path)
    print(f"Annotated image saved: {out_path}")


save_annotated(screenshot.copy(), elements)

```

### test_screenshot.png

(Skipped: binary or unreadable file)


### test.py

```python
import pyautogui
import time

# element 37 bbox [200, 1793, 295, 1888]
x1, y1, x2, y2 = 205, 1800, 291, 1882
cx = (x1 + x2) // 4
cy = (y1 + y2) // 4

print(f"Center: ({cx}, {cy})")
print("Moving mouse in 3 seconds...")

pyautogui.moveTo(cx, cy, duration=0.5)
print("Mouse moved — check where it is on screen")

```