"""
Generate text with Qwen3-VL and capture cross-attention to visual tokens.

Strategy:
- Load model with attn_implementation="eager" (required for output_attentions).
- With 504px cap, ViT has only ~640 patches -> eager attention is fast.
- Pass 1: model.generate() for token generation.
- Pass 2: forward with output_attentions=True for attention maps.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image


MAX_IMAGE_LONGEST_SIDE = 504


def _find_vision_token_range(input_ids: torch.Tensor, tokenizer) -> Tuple[int, int]:
    """Find the start and end indices of visual tokens in input_ids."""
    ids = input_ids[0].tolist()
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    start_idx = None
    end_idx = None
    for i, tid in enumerate(ids):
        if tid == start_id:
            start_idx = i + 1
        if tid == end_id:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Could not find vision token markers in input_ids")
    return start_idx, end_idx


def resize_for_model(image: Image.Image, max_side: int = MAX_IMAGE_LONGEST_SIDE) -> Image.Image:
    """Resize so longest side <= max_side, dims are multiples of 32 (for Qwen3-VL)."""
    w, h = image.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        w, h = int(w * scale), int(h * scale)
    w = (w // 32) * 32
    h = (h // 32) * 32
    w = max(32, w)
    h = max(32, h)
    return image.resize((w, h), Image.LANCZOS)


def generate_with_attention(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 30,
    layers: Optional[List[int]] = None,
) -> Tuple[str, np.ndarray]:
    """
    Generate answer and capture attention from answer tokens to visual tokens.

    Two-pass approach:
    1. generate() for fast answer.
    2. Forward pass with output_attentions=True for attention maps.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    vision_start, vision_end = _find_vision_token_range(input_ids, processor.tokenizer)
    num_visual_tokens = vision_end - vision_start
    print(f"  Visual tokens: {num_visual_tokens}, vision range: [{vision_start}:{vision_end}]")

    if layers is None:
        layers = list(range(12, 18))

    # Pass 1: Generate answer
    print("  Generating answer...")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = input_ids.shape[1]
    generated_ids = generated[0, input_len:]
    answer_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_gen_tokens = len(generated_ids)
    print(f"  Generated {num_gen_tokens} tokens: {answer_text}")

    # Pass 2: Forward with output_attentions=True
    print("  Extracting attention maps...")
    full_ids = generated[:, :input_len + num_gen_tokens]

    fwd_inputs = {}
    fwd_inputs["input_ids"] = full_ids.to(device)
    fwd_inputs["output_attentions"] = True
    fwd_inputs["use_cache"] = False

    for key in ["pixel_values", "image_grid_thw", "mm_token_type_ids"]:
        if key in inputs and inputs[key] is not None:
            val = inputs[key]
            if key == "mm_token_type_ids":
                pad = torch.zeros(1, num_gen_tokens, dtype=val.dtype, device=device)
                val = torch.cat([val, pad], dim=1)
            fwd_inputs[key] = val

    with torch.no_grad():
        outputs = model(**fwd_inputs)

    attentions = outputs.attentions
    attn_per_token = []

    for token_pos in range(input_len, input_len + num_gen_tokens):
        collected = []
        for layer_idx in layers:
            layer_attn = attentions[layer_idx][0, :, token_pos, vision_start:vision_end]
            layer_attn = layer_attn.mean(dim=0)
            collected.append(layer_attn.cpu().float().numpy())
        attn_per_token.append(np.mean(collected, axis=0))

    attn_matrix = np.stack(attn_per_token, axis=0)  # [num_gen_tokens, num_visual_tokens]
    return answer_text, attn_matrix
