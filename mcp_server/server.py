"""
GoClick MCP Server — exposes GUI grounding as MCP tools.

Tools:
  - goclick_point: Given a screenshot path + instruction, returns (x, y) coordinates.
  - goclick_point_base64: Same but accepts base64-encoded image.
  - goclick_health: Check if the model is loaded.

Run standalone:
    python -m mcp_server.server
"""

import base64
import io
import json
import re
import sys
import time

import torch
from PIL import Image
from fastmcp import FastMCP

sys.path.insert(0, ".")
import config  # noqa: E402 — sets HF_HOME

mcp = FastMCP("GoClick Grounding Server")

_model = None
_processor = None


def _ensure_model():
    global _model, _processor
    if _model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"Loading {config.GOCLICK_MODEL}...")
    _model = AutoModelForCausalLM.from_pretrained(
        config.GOCLICK_MODEL, trust_remote_code=True, torch_dtype=torch.float32,
    )
    _processor = AutoProcessor.from_pretrained(config.GOCLICK_MODEL, trust_remote_code=True)
    _model.eval()
    print("GoClick-Base loaded.")


def _parse_location(text: str):
    m = re.search(r"<loc_(\d+)>.*?<loc_(\d+)>", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _get_screen_scale():
    """Get the Retina scale factor (screencapture pixels / display points)."""
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        # Look for resolution line like "Resolution: 2880 x 1864 Retina"
        for line in result.stdout.splitlines():
            if "Resolution" in line and "Retina" in line:
                return 2.0
    except Exception:
        pass
    return 1.0


_screen_scale = None


def _run_inference(image: Image.Image, instruction: str) -> dict:
    global _screen_scale
    _ensure_model()
    orig_w, orig_h = image.size

    if _screen_scale is None:
        _screen_scale = _get_screen_scale()

    prompt = (
        f"I want to {instruction}. Please locate the target element "
        f"I should interact with. (Output the center coordinates of the target)"
    )

    inputs = _processor(images=image, text=prompt, return_tensors="pt")
    inputs = {
        k: v.to(_model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=50)
    elapsed = time.time() - t0

    response = _processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    loc_x, loc_y = _parse_location(response)

    if loc_x is None:
        return {"success": False, "error": "Could not parse coordinates"}

    pixel_x = loc_x / 999.0 * orig_w
    pixel_y = loc_y / 999.0 * orig_h

    point_x = pixel_x / _screen_scale
    point_y = pixel_y / _screen_scale

    return {
        "success": True,
        "x": round(point_x, 1),
        "y": round(point_y, 1),
        "image_size": {"width": orig_w, "height": orig_h},
        "scale_factor": _screen_scale,
        "inference_time": round(elapsed, 2),
    }


@mcp.tool()
def goclick_point(image_path: str, instruction: str) -> str:
    """
    Find the click target for an instruction on a screenshot.

    Args:
        image_path: Path to the screenshot image file.
        instruction: What to do, e.g. "click on Chrome" or "open the search bar".

    Returns:
        JSON with x, y pixel coordinates of where to click.
    """
    image = Image.open(image_path).convert("RGB")
    return json.dumps(_run_inference(image, instruction))


@mcp.tool()
def goclick_point_base64(image_base64: str, instruction: str) -> str:
    """
    Find the click target for an instruction on a base64-encoded screenshot.

    Args:
        image_base64: Base64-encoded PNG/JPEG screenshot.
        instruction: What to do.

    Returns:
        JSON with x, y pixel coordinates.
    """
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return json.dumps(_run_inference(image, instruction))


@mcp.tool()
def goclick_health() -> str:
    """Check if GoClick model is loaded and ready."""
    return json.dumps({
        "status": "ready" if _model is not None else "not_loaded",
        "model": config.GOCLICK_MODEL,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8222)
    args = parser.parse_args()

    print("Starting GoClick MCP Server...")
    _ensure_model()
    if args.transport == "sse":
        mcp.run(transport="sse", host="127.0.0.1", port=args.port)
    else:
        mcp.run(transport="stdio")
