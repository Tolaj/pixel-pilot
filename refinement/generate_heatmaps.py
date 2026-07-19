"""
Generate attention heatmaps for all ScreenSpot samples.

Runs each (image, instruction) pair through Qwen3-VL attention pipeline
and saves the resulting heatmap + metadata.

Usage:
    python -m refinement.generate_heatmaps
"""

import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from grounding.attention import generate_with_attention, resize_for_model
from grounding.patch_map import build_patch_map
from grounding.localize import attention_to_point

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"
DATA_DIR = Path("refinement/data/screenspot")
OUTPUT_DIR = Path("refinement/data/heatmaps")


def main():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Load annotations
    annotations_path = DATA_DIR / "annotations.json"
    if not annotations_path.exists():
        print("ERROR: Run 'python -m refinement.download_data' first.")
        return

    with open(annotations_path) as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations")

    # Load model
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total_start = time.time()

    for i, ann in enumerate(annotations):
        img_path = DATA_DIR / "images" / ann["image"]
        img = Image.open(img_path).convert("RGB")
        instruction = ann["instruction"]
        true_click = ann["click_point"]  # absolute pixels in original image

        # Resize for model
        resized = resize_for_model(img)
        resized_size = (resized.height, resized.width)
        original_size = (img.height, img.width)

        # Build prompt
        prompt = f"Look at this screenshot. Where is {instruction}? Describe its location briefly."

        try:
            # Generate attention
            answer, attn = generate_with_attention(
                model, processor, resized, prompt, max_new_tokens=30
            )

            # Get grid info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": resized},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[resized], return_tensors="pt")
            grid_thw = inputs["image_grid_thw"][0].tolist()

            patch_fn, grid_shape = build_patch_map(
                image_grid_thw=tuple(grid_thw),
                resized_size=resized_size,
                original_size=original_size,
                retina_scale=1.0,
                patch_size=16,
                spatial_merge_size=2,
            )

            # Get predicted point
            centroid, argmax_pt, heatmap = attention_to_point(attn, patch_fn, grid_shape)

            # Save heatmap as numpy array
            heatmap_path = OUTPUT_DIR / f"{i:04d}.npy"
            np.save(heatmap_path, heatmap)

            # Compute error
            dist = np.sqrt(
                (centroid[0] - true_click[0])**2 +
                (centroid[1] - true_click[1])**2
            )

            results.append({
                "id": i,
                "instruction": instruction,
                "true_click": true_click,
                "predicted_click": [centroid[0], centroid[1]],
                "argmax_click": [argmax_pt[0], argmax_pt[1]],
                "distance_px": dist,
                "heatmap_file": f"{i:04d}.npy",
                "heatmap_shape": list(grid_shape),
                "image_size": ann["image_size"],
                "resized_size": list(resized_size),
                "answer": answer,
                "data_source": ann["data_source"],
                "data_type": ann["data_type"],
            })

            status = "PASS" if dist < 50 else "FAIL"
            print(f"[{i+1}/{len(annotations)}] {status} dist={dist:.0f}px | {instruction}")

        except Exception as e:
            print(f"[{i+1}/{len(annotations)}] ERROR: {e} | {instruction}")
            results.append({
                "id": i,
                "instruction": instruction,
                "error": str(e),
            })

        # Save intermediate results every 50 samples
        if (i + 1) % 50 == 0:
            _save_results(results)

    total_time = time.time() - total_start
    _save_results(results)

    # Print summary
    valid = [r for r in results if "distance_px" in r]
    if valid:
        dists = [r["distance_px"] for r in valid]
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(annotations)}")
        print(f"Successful: {len(valid)}")
        print(f"Errors: {len(results) - len(valid)}")
        print(f"Mean distance: {np.mean(dists):.1f}px")
        print(f"Median distance: {np.median(dists):.1f}px")
        print(f"Pass (<50px): {sum(1 for d in dists if d < 50)}/{len(valid)} ({100*sum(1 for d in dists if d < 50)/len(valid):.1f}%)")
        print(f"Total time: {total_time:.0f}s ({total_time/len(annotations):.1f}s/sample)")


def _save_results(results):
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
