"""
Parameter sweep: token selection, temperature, top-k fraction.

Usage:
    python -m benchmarks.tune_params
"""

import torch
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"


def create_test_cases():
    cases = []

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([360, 180, 420, 240], fill="red")
    cases.append(("red_sq_BR", img, "the red square", (390, 210)))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 50, 130, 110], fill="red")
    cases.append(("red_sq_TL", img, "the red square", (100, 80)))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([216, 120, 296, 200], fill="blue")
    cases.append(("blue_circle", img, "the blue circle", (256, 160)))

    img = Image.new("RGB", (512, 320), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([350, 250, 480, 290], radius=8, fill=(34, 139, 34))
    draw.text((380, 260), "Submit", fill="white")
    cases.append(("green_btn", img, "the green Submit button", (415, 270)))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 130, 260, 190], fill="red")
    draw.rectangle([50, 50, 90, 90], fill="blue")
    draw.ellipse([400, 220, 460, 280], fill="green")
    draw.rectangle([380, 30, 420, 70], fill="yellow")
    cases.append(("red_sq_distract", img, "the red square", (230, 160)))

    return cases


def main():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from grounding.attention import generate_with_attention
    from grounding.patch_map import build_patch_map

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    cases = create_test_cases()
    prompt_template = "Look at this screenshot. Where is {target}? Describe its location briefly."

    print("\nGenerating attention matrices for all test cases...")
    case_data = []
    for name, img, target, true_center in cases:
        prompt = prompt_template.format(target=target)
        answer, attn = generate_with_attention(model, processor, img, prompt, max_new_tokens=30)

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

        case_data.append({
            "name": name, "true_center": true_center,
            "attn": attn, "patch_fn": patch_fn,
            "grid_shape": grid_shape, "answer": answer,
        })
        print(f"  {name}: '{answer}'")

    # Sweep
    token_configs = {
        "tok_1-2": (1, 2),
        "tok_1-3": (1, 3),
        "tok_0-3": (0, 3),
        "tok_1-4": (1, 4),
        "tok_0-2": (0, 2),
        "all": None,
    }
    temp_configs = [0.05, 0.1, 0.2]
    topk_configs = [0.03, 0.05, 0.10]

    print("\nSweeping token selection, temperature, top-k...")
    print(f"{'Config':<35} {'Mean':>6} {'Max':>6} {'Pass':>5}  Per-case distances")
    print("-" * 100)

    results_table = []
    for tok_name, tok_range in token_configs.items():
        for temp in temp_configs:
            for topk in topk_configs:
                distances = []
                for cd in case_data:
                    attn = cd["attn"]
                    if tok_range is not None:
                        start, count = tok_range
                        end = min(start + count, attn.shape[0])
                        agg = attn[start:end].mean(axis=0)
                    else:
                        agg = attn.mean(axis=0)

                    flat = agg / temp
                    flat = flat - flat.max()
                    weights = np.exp(flat)
                    weights = weights / weights.sum()

                    k = max(1, int(len(weights) * topk))
                    threshold = np.sort(weights)[-k]
                    mask = weights >= threshold
                    weights = weights * mask
                    weights = weights / weights.sum()

                    cx, cy = 0.0, 0.0
                    for i in range(len(weights)):
                        if weights[i] > 0:
                            sx, sy = cd["patch_fn"](i)
                            cx += weights[i] * sx
                            cy += weights[i] * sy

                    dist = np.sqrt((cx - cd["true_center"][0])**2 + (cy - cd["true_center"][1])**2)
                    distances.append(dist)

                mean_dist = np.mean(distances)
                max_dist = np.max(distances)
                passed = sum(1 for d in distances if d < 80)
                config_name = f"{tok_name} t={temp} k={topk}"
                dist_str = "  ".join(f"{d:5.0f}" for d in distances)
                marker = " <<<" if passed >= 4 else (" <<" if passed >= 3 else "")
                print(f"{config_name:<35} {mean_dist:6.1f} {max_dist:6.1f} {passed:>3}/5  {dist_str}{marker}")

                results_table.append((config_name, mean_dist, max_dist, passed, distances))

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    results_table.sort(key=lambda x: (-x[3], x[1]))
    for i, (config, mean_d, max_d, passed, dists) in enumerate(results_table[:10]):
        dist_str = "  ".join(f"{d:5.0f}" for d in dists)
        print(f"  {i+1}. {config:<35} pass={passed}/5  mean={mean_d:.1f}px  max={max_d:.1f}px  [{dist_str}]")


if __name__ == "__main__":
    main()
