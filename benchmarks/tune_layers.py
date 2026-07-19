"""
Sweep layer selection for attention grounding.

Usage:
    python -m benchmarks.tune_layers
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
    from grounding.attention import _find_vision_token_range
    from grounding.patch_map import build_patch_map

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    device = next(model.parameters()).device

    cases = create_test_cases()
    prompt_template = "Look at this screenshot. Where is {target}? Describe its location briefly."

    print("\nGenerating full per-layer attention...")
    case_data = []
    for name, img, target, true_center in cases:
        prompt = prompt_template.format(target=target)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        vision_start, vision_end = _find_vision_token_range(input_ids, processor.tokenizer)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_len = input_ids.shape[1]
        gen_ids = generated[0, input_len:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        tokens = [processor.tokenizer.decode([t]) for t in gen_ids]
        num_gen = len(gen_ids)

        full_ids = generated[:, :input_len + num_gen]
        fwd_inputs = {"input_ids": full_ids, "output_attentions": True, "use_cache": False}
        for key in ["pixel_values", "image_grid_thw", "mm_token_type_ids"]:
            if key in inputs and inputs[key] is not None:
                val = inputs[key]
                if key == "mm_token_type_ids":
                    pad = torch.zeros(1, num_gen, dtype=val.dtype, device=device)
                    val = torch.cat([val, pad], dim=1)
                fwd_inputs[key] = val

        with torch.no_grad():
            outputs = model(**fwd_inputs)

        num_layers = len(outputs.attentions)
        per_layer_attn = []
        for li in range(num_layers):
            token_attns = []
            for tp in range(input_len, input_len + num_gen):
                a = outputs.attentions[li][0, :, tp, vision_start:vision_end]
                token_attns.append(a.mean(dim=0).cpu().float().numpy())
            per_layer_attn.append(np.stack(token_attns))
        per_layer_attn = np.stack(per_layer_attn)

        grid_thw = inputs["image_grid_thw"][0].tolist()
        patch_fn, grid_shape = build_patch_map(
            image_grid_thw=tuple(grid_thw),
            resized_size=(img.height, img.width),
            original_size=(img.height, img.width),
            retina_scale=1.0, patch_size=16, spatial_merge_size=2,
        )

        case_data.append({
            "name": name, "true_center": true_center,
            "per_layer_attn": per_layer_attn,
            "patch_fn": patch_fn, "grid_shape": grid_shape,
            "tokens": tokens, "answer": answer,
        })
        print(f"  {name}: '{answer}'")

    tok_start, tok_count = 1, 2
    temp, topk = 0.05, 0.05

    layer_configs = {}
    for i in range(28):
        layer_configs[f"L{i}"] = [i]
    for start in range(0, 25, 2):
        for width in [4, 6, 8]:
            end = min(start + width, 28)
            layer_configs[f"L{start}-{end}"] = list(range(start, end))

    print(f"\nSweeping layers (tok=1-2, t={temp}, k={topk})")
    print(f"{'Config':<15} {'Mean':>6} {'Max':>6} {'Pass':>5}  Per-case distances")
    print("-" * 90)

    results = []
    for lname, lidx in layer_configs.items():
        distances = []
        for cd in case_data:
            attn = cd["per_layer_attn"][lidx].mean(axis=0)
            end = min(tok_start + tok_count, attn.shape[0])
            agg = attn[tok_start:end].mean(axis=0)

            flat = agg / temp
            flat = flat - flat.max()
            weights = np.exp(flat)
            weights /= weights.sum()
            k = max(1, int(len(weights) * topk))
            threshold = np.sort(weights)[-k]
            mask = weights >= threshold
            weights = weights * mask
            weights /= weights.sum()

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
        results.append((lname, mean_dist, max_dist, passed, distances))

    results.sort(key=lambda x: (-x[3], x[1]))
    for lname, mean_d, max_d, passed, dists in results[:20]:
        dist_str = "  ".join(f"{d:5.0f}" for d in dists)
        marker = " <<<" if passed >= 4 else ""
        print(f"{lname:<15} {mean_d:6.1f} {max_d:6.1f} {passed:>3}/5  {dist_str}{marker}")


if __name__ == "__main__":
    main()
