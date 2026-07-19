"""Convert aggregated attention heatmap to a screen coordinate."""

import numpy as np
from typing import Tuple, Callable
from pathlib import Path
from PIL import Image


def attention_to_point(
    attn: np.ndarray,
    patch_map: Callable[[int], Tuple[float, float]],
    grid_shape: Tuple[int, int],
    temperature: float = 0.05,
    top_k_frac: float = 0.05,
    mode: str = "object_tokens",
    object_token_count: int = 2,
) -> Tuple[Tuple[float, float], Tuple[float, float], np.ndarray]:
    """
    Convert attention matrix to a screen point.

    Args:
        attn: shape [num_generated_tokens, num_visual_tokens]
        patch_map: token_idx -> (screen_x, screen_y)
        grid_shape: (rows, cols) of merged patch grid
        temperature: softmax temperature for sharpening (lower = sharper)
        top_k_frac: fraction of patches to keep (top attention)
        mode: "object_tokens" uses first N tokens; "mean" averages all.
        object_token_count: how many initial tokens to use in object_tokens mode.

    Returns:
        (centroid_point, argmax_point, heatmap_2d)
    """
    rows, cols = grid_shape

    if mode == "object_tokens":
        start = 1
        end = min(start + object_token_count, attn.shape[0])
        agg = attn[start:end].mean(axis=0)
    else:
        agg = attn.mean(axis=0)

    heatmap = agg.reshape(rows, cols)

    # Softmax sharpening
    flat = agg / temperature
    flat = flat - flat.max()
    weights = np.exp(flat)
    weights = weights / weights.sum()

    # Top-k masking
    k = max(1, int(len(weights) * top_k_frac))
    threshold = np.sort(weights)[-k]
    mask = weights >= threshold
    weights = weights * mask
    weights = weights / weights.sum()

    # Weighted centroid
    cx, cy = 0.0, 0.0
    for i in range(len(weights)):
        if weights[i] > 0:
            sx, sy = patch_map(i)
            cx += weights[i] * sx
            cy += weights[i] * sy

    # Argmax point
    argmax_idx = int(np.argmax(agg))
    argmax_point = patch_map(argmax_idx)

    # Normalize heatmap for visualization
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return (cx, cy), argmax_point, heatmap_norm


def save_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    point: Tuple[float, float],
    path: str,
    retina_scale: float = 2.0,
) -> None:
    """Save a debug overlay: original image with heatmap and predicted click point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    img_w, img_h = image.size
    ax.imshow(
        heatmap,
        extent=[0, img_w, img_h, 0],
        alpha=0.4,
        cmap="jet",
        interpolation="bilinear",
    )

    px = point[0] * retina_scale
    py = point[1] * retina_scale
    circle = Circle((px, py), radius=15, color="lime", fill=False, linewidth=3)
    ax.add_patch(circle)
    ax.plot(px, py, "x", color="lime", markersize=15, markeredgewidth=3)

    ax.set_title(f"Predicted: ({point[0]:.0f}, {point[1]:.0f}) screen points")
    ax.axis("off")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
