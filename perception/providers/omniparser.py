"""OmniParser v2: YOLO icon detection + Florence-2 captioning."""

import os
import torch
from PIL import Image
from .base_provider import LazyProvider
from ..registry import register

os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

OMNI_DIR = "./models/omniparser"
MIN_CONF = 0.5
ICON_DETECT = os.path.join(OMNI_DIR, "icon_detect/model.pt")
ICON_CAPTION = os.path.join(OMNI_DIR, "icon_caption_florence")
FLORENCE_SAFE = os.path.join(ICON_CAPTION, "model.safetensors")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16


@register
class OmniParserProvider(LazyProvider):
    name = "omniparser"
    description = (
        "OmniParser v2: YOLO icon detection + Florence-2 captioning (~2-5s/frame)"
    )

    def _load(self):
        from ultralytics import YOLO
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not os.path.exists(FLORENCE_SAFE):
            print("[omniparser] Converting Florence-2 weights (one-time ~30s)...")
            tmp = AutoModelForCausalLM.from_pretrained(
                ICON_CAPTION, torch_dtype=DTYPE, trust_remote_code=True
            )
            tmp.save_pretrained(ICON_CAPTION)
            del tmp
            if DEVICE == "mps":
                torch.mps.empty_cache()
            print("[omniparser] Conversion done.")

        print("[omniparser] Loading YOLO (cpu)...")
        self._yolo = YOLO(ICON_DETECT)
        self._yolo.to("cpu")

        print(f"[omniparser] Loading Florence-2 ({DEVICE})...")
        self._processor = AutoProcessor.from_pretrained(
            ICON_CAPTION, trust_remote_code=True
        )
        self._florence = AutoModelForCausalLM.from_pretrained(
            ICON_CAPTION,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(DEVICE)
        self._florence.eval()

    def _caption_batch(self, crops: list, batch_size: int = 32) -> list:
        results = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            inputs = self._processor(
                text=["<CAPTION>"] * len(batch),
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE, DTYPE)
            with torch.inference_mode():
                ids = self._florence.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20,
                    num_beams=1,
                )
            decoded = self._processor.batch_decode(ids, skip_special_tokens=True)
            results.extend([s.strip() or "icon" for s in decoded])
        return results

    def parse(self, image: Image.Image) -> list[dict]:
        self._ensure_loaded()
        image_rgb = image.convert("RGB")

        boxes = self._yolo(image_rgb, verbose=False)[0].boxes
        kept, crops = [], []
        for box in boxes:
            conf = box.conf[0].item()
            if conf < MIN_CONF:
                continue
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            kept.append((x1, y1, x2, y2, round(conf, 2)))
            crops.append(image_rgb.crop((x1, y1, x2, y2)))

        captions = self._caption_batch(crops) if crops else []

        elements = []
        for (x1, y1, x2, y2, conf), label in zip(kept, captions):
            elements.append(
                {
                    "id": len(elements),
                    "label": label,
                    "role": "unknown",
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "source": self.name,
                }
            )
        return elements

    def teardown(self):
        del self._yolo, self._florence, self._processor
        if DEVICE == "mps":
            torch.mps.empty_cache()
