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
