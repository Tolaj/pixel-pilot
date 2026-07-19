"""
Synthetic grounding tests: verify attention peaks on known targets.

Requires model downloaded. Run with:
    pytest tests/test_synthetic_grounding.py -v -s -m slow
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"


def create_synthetic_image(
    width: int = 512,
    height: int = 320,
    square_center: tuple = (380, 200),
    square_size: int = 60,
) -> tuple:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    x, y = square_center
    half = square_size // 2
    draw.rectangle([x - half, y - half, x + half, y + half], fill="red")
    return img, square_center


@pytest.fixture(scope="module")
def model_and_processor():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    return model, processor


@pytest.mark.slow
class TestSyntheticGrounding:

    def test_red_square_grounding(self, model_and_processor):
        from grounding.attention import generate_with_attention
        from grounding.patch_map import build_patch_map
        from grounding.localize import attention_to_point

        model, processor = model_and_processor

        img, true_center = create_synthetic_image(512, 320, (380, 200), 60)

        prompt = "Look at this screenshot. Where is the red square? Describe its location briefly."
        answer, attn = generate_with_attention(model, processor, img, prompt, max_new_tokens=30)
        print(f"Model answer: {answer}")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        grid_thw = inputs["image_grid_thw"][0].tolist()

        patch_fn, grid_shape = build_patch_map(
            image_grid_thw=tuple(grid_thw),
            resized_size=(img.height, img.width),
            original_size=(img.height, img.width),
            retina_scale=1.0, patch_size=16, spatial_merge_size=2,
        )

        centroid, argmax_pt, heatmap = attention_to_point(attn, patch_fn, grid_shape)
        print(f"True: {true_center}, Predicted: ({centroid[0]:.0f}, {centroid[1]:.0f})")

        dist = np.sqrt((centroid[0] - true_center[0])**2 + (centroid[1] - true_center[1])**2)
        print(f"Distance: {dist:.1f}px")
        assert dist < 80

    def test_red_square_different_position(self, model_and_processor):
        from grounding.attention import generate_with_attention
        from grounding.patch_map import build_patch_map
        from grounding.localize import attention_to_point

        model, processor = model_and_processor

        img, true_center = create_synthetic_image(512, 320, (100, 80), 60)

        prompt = "Look at this screenshot. Where is the red square? Describe its location briefly."
        answer, attn = generate_with_attention(model, processor, img, prompt, max_new_tokens=30)
        print(f"Model answer: {answer}")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        grid_thw = inputs["image_grid_thw"][0].tolist()

        patch_fn, grid_shape = build_patch_map(
            image_grid_thw=tuple(grid_thw),
            resized_size=(img.height, img.width),
            original_size=(img.height, img.width),
            retina_scale=1.0, patch_size=16, spatial_merge_size=2,
        )

        centroid, argmax_pt, heatmap = attention_to_point(attn, patch_fn, grid_shape)
        print(f"True: {true_center}, Predicted: ({centroid[0]:.0f}, {centroid[1]:.0f})")

        dist = np.sqrt((centroid[0] - true_center[0])**2 + (centroid[1] - true_center[1])**2)
        print(f"Distance: {dist:.1f}px")
        assert dist < 80
