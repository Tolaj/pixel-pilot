"""
Attention grounding benchmark with visual HTML report.

Usage:
    python -m benchmarks.run
"""

import time
import base64
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = "./models/Qwen3-VL-2B-Instruct"
REPORT_DIR = Path("benchmarks/results")


@dataclass
class TestCase:
    name: str
    description: str
    image: Image.Image
    target: str
    true_center: Tuple[int, int]
    prompt: str = ""

    def __post_init__(self):
        if not self.prompt:
            self.prompt = f"Look at this screenshot. Where is {self.target}? Describe its location briefly."


@dataclass
class TestResult:
    case: TestCase
    answer: str
    centroid: Tuple[float, float]
    argmax_point: Tuple[float, float]
    distance_px: float
    time_generate_s: float
    time_attention_s: float
    heatmap_overlay: Image.Image
    passed: bool
    threshold_px: float = 80.0


def create_test_cases() -> List[TestCase]:
    """Create synthetic test cases with known ground truth."""
    cases = []

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([360, 180, 420, 240], fill="red")
    cases.append(TestCase(
        name="red_square_bottom_right",
        description="Red square at (390, 210) on white background",
        image=img, target="the red square", true_center=(390, 210),
    ))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 50, 130, 110], fill="red")
    cases.append(TestCase(
        name="red_square_top_left",
        description="Red square at (100, 80) on white background",
        image=img, target="the red square", true_center=(100, 80),
    ))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([216, 120, 296, 200], fill="blue")
    cases.append(TestCase(
        name="blue_circle_center",
        description="Blue circle at (256, 160) on white background",
        image=img, target="the blue circle", true_center=(256, 160),
    ))

    img = Image.new("RGB", (512, 320), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([350, 250, 480, 290], radius=8, fill=(34, 139, 34))
    draw.text((380, 260), "Submit", fill="white")
    cases.append(TestCase(
        name="green_button",
        description="Green 'Submit' button at (415, 270) on gray background",
        image=img, target="the green Submit button", true_center=(415, 270),
    ))

    img = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 130, 260, 190], fill="red")
    draw.rectangle([50, 50, 90, 90], fill="blue")
    draw.ellipse([400, 220, 460, 280], fill="green")
    draw.rectangle([380, 30, 420, 70], fill="yellow")
    cases.append(TestCase(
        name="red_square_with_distractors",
        description="Red square at (230, 160) with blue/green/yellow distractors",
        image=img, target="the red square", true_center=(230, 160),
    ))

    return cases


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    centroid: Tuple[float, float],
    true_center: Tuple[int, int],
) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.imshow(image)

    img_w, img_h = image.size
    ax.imshow(
        heatmap, extent=[0, img_w, img_h, 0],
        alpha=0.4, cmap="jet", interpolation="bilinear",
    )

    ax.plot(centroid[0], centroid[1], "x", color="lime", markersize=20, markeredgewidth=3)
    circle = Circle(centroid, radius=12, color="lime", fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.plot(true_center[0], true_center[1], "+", color="red", markersize=20, markeredgewidth=3)
    circle = Circle(true_center, radius=12, color="red", fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.set_title(f"Predicted (green): ({centroid[0]:.0f}, {centroid[1]:.0f})  |  True (red): {true_center}", fontsize=10)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def generate_html_report(results: List[TestResult], total_time: float) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    rows = []
    for r in results:
        input_b64 = image_to_base64(r.case.image)
        overlay_b64 = image_to_base64(r.heatmap_overlay)
        status = "PASS" if r.passed else "FAIL"
        status_color = "#22c55e" if r.passed else "#ef4444"

        rows.append(f"""
        <div class="test-card">
            <div class="card-header" style="border-left: 4px solid {status_color}">
                <span class="status" style="color: {status_color}">{status}</span>
                <span class="name">{r.case.name}</span>
                <span class="distance">{r.distance_px:.1f}px</span>
            </div>
            <div class="card-body">
                <div class="images">
                    <div class="img-container">
                        <div class="img-label">Input</div>
                        <img src="data:image/png;base64,{input_b64}" />
                    </div>
                    <div class="img-container">
                        <div class="img-label">Attention Heatmap</div>
                        <img src="data:image/png;base64,{overlay_b64}" />
                    </div>
                </div>
                <div class="details">
                    <p><strong>Target:</strong> {r.case.target}</p>
                    <p><strong>Model answer:</strong> {r.answer}</p>
                    <p><strong>True center:</strong> {r.case.true_center}</p>
                    <p><strong>Predicted (centroid):</strong> ({r.centroid[0]:.0f}, {r.centroid[1]:.0f})</p>
                    <p><strong>Predicted (argmax):</strong> ({r.argmax_point[0]:.0f}, {r.argmax_point[1]:.0f})</p>
                    <p><strong>Distance:</strong> {r.distance_px:.1f}px (threshold: {r.threshold_px:.0f}px)</p>
                    <p><strong>Time:</strong> generate {r.time_generate_s:.1f}s + attention {r.time_attention_s:.1f}s</p>
                </div>
            </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Attention Grounding Benchmark</title>
<style>
    body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
    h1 {{ color: #fff; margin-bottom: 5px; }}
    .summary {{ background: #16213e; padding: 16px 24px; border-radius: 8px; margin-bottom: 24px; display: flex; gap: 32px; align-items: center; }}
    .summary .stat {{ text-align: center; }}
    .summary .stat .value {{ font-size: 28px; font-weight: bold; }}
    .summary .stat .label {{ font-size: 12px; opacity: 0.7; text-transform: uppercase; }}
    .pass-rate {{ color: {"#22c55e" if passed == total else "#ef4444"}; }}
    .test-card {{ background: #16213e; border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
    .card-header {{ padding: 12px 16px; display: flex; align-items: center; gap: 12px; background: #0f3460; }}
    .card-header .status {{ font-weight: bold; font-size: 14px; }}
    .card-header .name {{ flex: 1; font-family: monospace; }}
    .card-header .distance {{ font-family: monospace; opacity: 0.7; }}
    .card-body {{ padding: 16px; }}
    .images {{ display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }}
    .img-container {{ flex: 1; min-width: 280px; }}
    .img-container img {{ width: 100%; border-radius: 4px; border: 1px solid #333; }}
    .img-label {{ font-size: 11px; text-transform: uppercase; opacity: 0.6; margin-bottom: 4px; }}
    .details {{ font-size: 13px; line-height: 1.6; }}
    .details p {{ margin: 2px 0; }}
    .model-info {{ font-size: 12px; opacity: 0.6; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>Attention Grounding Benchmark</h1>
<p class="model-info">Model: Qwen3-VL-2B-Instruct | Layers: 12-18 | Tokens: 1-2 (object name) | Temp: 0.05, Top-k: 5% | Image cap: 504px</p>

<div class="summary">
    <div class="stat">
        <div class="value pass-rate">{passed}/{total}</div>
        <div class="label">Tests Passed</div>
    </div>
    <div class="stat">
        <div class="value">{np.mean([r.distance_px for r in results]):.0f}px</div>
        <div class="label">Mean Distance</div>
    </div>
    <div class="stat">
        <div class="value">{total_time:.0f}s</div>
        <div class="label">Total Time</div>
    </div>
</div>

{"".join(rows)}
</body>
</html>"""
    return html


def run():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from grounding.attention import generate_with_attention
    from grounding.patch_map import build_patch_map
    from grounding.localize import attention_to_point

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.\n")

    cases = create_test_cases()
    results = []
    total_start = time.time()

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case.name}: {case.description}")

        t0 = time.time()
        answer, attn = generate_with_attention(
            model, processor, case.image, case.prompt, max_new_tokens=30
        )
        t_gen = time.time() - t0

        messages = [{"role": "user", "content": [
            {"type": "image", "image": case.image},
            {"type": "text", "text": case.prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[case.image], return_tensors="pt")
        grid_thw = inputs["image_grid_thw"][0].tolist()

        patch_fn, grid_shape = build_patch_map(
            image_grid_thw=tuple(grid_thw),
            resized_size=(case.image.height, case.image.width),
            original_size=(case.image.height, case.image.width),
            retina_scale=1.0, patch_size=16, spatial_merge_size=2,
        )

        t1 = time.time()
        centroid, argmax_pt, heatmap = attention_to_point(attn, patch_fn, grid_shape)
        t_attn = time.time() - t1

        dist = float(np.sqrt(
            (centroid[0] - case.true_center[0])**2 +
            (centroid[1] - case.true_center[1])**2
        ))

        overlay = create_heatmap_overlay(case.image, heatmap, centroid, case.true_center)

        result = TestResult(
            case=case, answer=answer, centroid=centroid,
            argmax_point=argmax_pt, distance_px=dist,
            time_generate_s=t_gen, time_attention_s=t_attn,
            heatmap_overlay=overlay, passed=dist < 80,
        )
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} | dist={dist:.1f}px | answer: {answer}\n")

    total_time = time.time() - total_start

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_html = generate_html_report(results, total_time)
    report_path = REPORT_DIR / "report.html"
    report_path.write_text(report_html)
    print(f"\nReport saved: {report_path}")
    print(f"Total time: {total_time:.0f}s | Passed: {sum(1 for r in results if r.passed)}/{len(results)}")


if __name__ == "__main__":
    run()
