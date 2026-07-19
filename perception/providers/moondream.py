"""Moondream2 provider -- 1.8B VLM, fast screenshot understanding."""

import os
import torch
from PIL import Image
from .base_provider import LazyProvider
from ..registry import register

MOONDREAM_DIR = "./models/moondream2"
MOONDREAM_GRID_MODE = False

GRID_COLS = 4
GRID_ROWS = 4

OMNI_DIR = "./models/omniparser"
ICON_DETECT = os.path.join(OMNI_DIR, "icon_detect/model.pt")
MIN_CONF = 0.5

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16


@register
class MoondreamProvider(LazyProvider):
    name = "moondream"
    description = (
        "Moondream2 1.8B VLM -- faster than Qwen3-VL, no server needed (~500ms/frame)"
    )

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[moondream] Loading model ({DEVICE})...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            MOONDREAM_DIR, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MOONDREAM_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            attn_implementation="eager",
        ).to(DEVICE)
        self._model.eval()

        if not MOONDREAM_GRID_MODE:
            from ultralytics import YOLO

            print("[moondream] Loading YOLO for region detection (cpu)...")
            self._yolo = YOLO(ICON_DETECT)
            self._yolo.to("cpu")
        else:
            self._yolo = None

        print("[moondream] Ready.")

    def _ask(self, image: Image.Image, question: str) -> str:
        enc = self._model.encode_image(image)
        return self._model.answer_question(enc, question, self._tokenizer).strip()

    def _parse_grid(self, image: Image.Image) -> list[dict]:
        w, h = image.size
        cw, ch = w // GRID_COLS, h // GRID_ROWS
        elements = []

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x1 = col * cw
                y1 = row * ch
                x2 = x1 + cw
                y2 = y1 + ch
                crop = image.crop((x1, y1, x2, y2))
                label = self._ask(
                    crop, "What UI element or text is shown here? Be brief."
                )
                if label and label.lower() not in ("none", "empty", "blank", ""):
                    elements.append(
                        {
                            "id": len(elements),
                            "label": label,
                            "role": "unknown",
                            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                            "bbox": [x1, y1, x2, y2],
                            "conf": 0.8,
                            "source": self.name,
                        }
                    )
        return elements

    def _parse_yolo_guided(self, image: Image.Image) -> list[dict]:
        image_rgb = image.convert("RGB")
        boxes = self._yolo(image_rgb, verbose=False)[0].boxes
        elements = []

        for box in boxes:
            conf = box.conf[0].item()
            if conf < MIN_CONF:
                continue
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            crop = image_rgb.crop((x1, y1, x2, y2))
            label = self._ask(
                crop, "What is this UI element? Reply with a short label only."
            )
            elements.append(
                {
                    "id": len(elements),
                    "label": label or "icon",
                    "role": "unknown",
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 2),
                    "source": self.name,
                }
            )
        return elements

    def parse(self, image: Image.Image) -> list[dict]:
        self._ensure_loaded()
        if MOONDREAM_GRID_MODE:
            return self._parse_grid(image)
        return self._parse_yolo_guided(image)

    def teardown(self):
        del self._model, self._tokenizer
        if self._yolo:
            del self._yolo
        if DEVICE == "mps":
            torch.mps.empty_cache()
