"""
CLI for attention-based click grounding.

Usage:
    pixel-pilot --target "the Submit button" [--dry-run] [--image path.png]
    python cli.py --target "the red square"
"""

import argparse
import subprocess
import tempfile
import time

import numpy as np
import torch
from PIL import Image

from grounding.attention import generate_with_attention, resize_for_model
from grounding.patch_map import build_patch_map
from grounding.localize import attention_to_point, save_heatmap_overlay
from grounding.click import click


MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"


def take_screenshot() -> Image.Image:
    """Capture the current macOS screen."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    subprocess.run(["screencapture", "-x", tmp.name], check=True)
    return Image.open(tmp.name)


def load_model():
    """Load Qwen3-VL-2B-Instruct on MPS with fp16."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.")
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Attention-based click grounding")
    parser.add_argument("--target", required=True, help="Description of the UI element to click")
    parser.add_argument("--dry-run", action="store_true", help="Preview click without executing")
    parser.add_argument("--image", type=str, help="Path to screenshot (default: capture screen)")
    parser.add_argument("--temperature", type=float, default=0.05, help="Softmax temperature")
    parser.add_argument("--top-k-frac", type=float, default=0.05, help="Top-k fraction of patches")
    parser.add_argument("--retina-scale", type=float, default=2.0, help="Retina scaling factor")
    args = parser.parse_args()

    if args.image:
        original_image = Image.open(args.image).convert("RGB")
    else:
        original_image = take_screenshot().convert("RGB")

    original_size = (original_image.height, original_image.width)
    resized_image = resize_for_model(original_image)
    resized_size = (resized_image.height, resized_image.width)

    print(f"Original: {original_size[1]}x{original_size[0]}, Resized: {resized_size[1]}x{resized_size[0]}")

    model, processor = load_model()

    prompt = f"Look at this screenshot. Where is {args.target}? Describe its location briefly."

    print(f"Generating response for: '{args.target}'...")
    t0 = time.time()
    answer_text, attn_matrix = generate_with_attention(
        model, processor, resized_image, prompt, max_new_tokens=30
    )
    print(f"Answer ({time.time()-t0:.1f}s): {answer_text}")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": resized_image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[resized_image], return_tensors="pt")
    image_grid_thw = inputs["image_grid_thw"][0].tolist()

    patch_map_fn, grid_shape = build_patch_map(
        image_grid_thw=tuple(image_grid_thw),
        resized_size=resized_size,
        original_size=original_size,
        retina_scale=args.retina_scale,
        patch_size=16,
        spatial_merge_size=2,
    )

    centroid, argmax_pt, heatmap = attention_to_point(
        attn_matrix, patch_map_fn, grid_shape,
        temperature=args.temperature,
        top_k_frac=args.top_k_frac,
    )

    print(f"Centroid: ({centroid[0]:.0f}, {centroid[1]:.0f}), Argmax: ({argmax_pt[0]:.0f}, {argmax_pt[1]:.0f})")

    timestamp = int(time.time())
    debug_path = f"debug/heatmap_{timestamp}.png"
    save_heatmap_overlay(original_image, heatmap, centroid, debug_path, retina_scale=args.retina_scale)
    print(f"Heatmap overlay saved: {debug_path}")

    click(centroid[0], centroid[1], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
