"""
Per-token, per-layer attention argmax analysis.

Usage:
    python -m benchmarks.diagnose
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from grounding.attention import _find_vision_token_range

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"


def create_synthetic_image():
    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([350, 170, 410, 230], fill="red")
    return img


def main():
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    device = next(model.parameters()).device

    img = create_synthetic_image()

    prompts = [
        "Look at this screenshot. Where is the red square? Describe its location briefly.",
        "Point to the red square in the image.",
        "Focus on the red square.",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        vision_start, vision_end = _find_vision_token_range(input_ids, processor.tokenizer)
        num_vis = vision_end - vision_start

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        input_len = input_ids.shape[1]
        gen_ids = generated[0, input_len:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        tokens = [processor.tokenizer.decode([t]) for t in gen_ids]
        num_gen = len(gen_ids)
        print(f"Answer: {answer}")
        print(f"Tokens: {tokens}")

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

        grid_thw = inputs["image_grid_thw"][0].tolist()
        merged_cols = grid_thw[2] // 2

        def idx_to_xy(idx):
            row = idx // merged_cols
            col = idx % merged_cols
            return ((col + 0.5) * 32, (row + 0.5) * 32)

        num_layers = len(outputs.attentions)
        layer_groups = {
            "early (0-7)": list(range(0, 8)),
            "middle (8-19)": list(range(8, 20)),
            "late (20-27)": list(range(20, num_layers)),
            "L12-18": list(range(12, 18)),
        }

        for group_name, layer_indices in layer_groups.items():
            print(f"\n  Layer group: {group_name}")
            for tok_idx, tok_text in enumerate(tokens[:10]):
                token_pos = input_len + tok_idx
                collected = []
                for li in layer_indices:
                    attn = outputs.attentions[li][0, :, token_pos, vision_start:vision_end]
                    collected.append(attn.mean(dim=0).cpu().float().numpy())
                avg = np.mean(collected, axis=0)
                argmax_idx = int(np.argmax(avg))
                xy = idx_to_xy(argmax_idx)
                print(f"    Token '{tok_text}' -> argmax patch {argmax_idx} = ({xy[0]:.0f}, {xy[1]:.0f}), max_val={avg[argmax_idx]:.4f}")


if __name__ == "__main__":
    main()
