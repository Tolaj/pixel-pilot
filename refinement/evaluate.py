"""
Evaluate refinement CNN vs raw attention baseline.

Usage:
    python -m refinement.evaluate
"""

import json
from pathlib import Path

import numpy as np
import torch

from refinement.model import RefinementNet

OUTPUT_DIR = Path("refinement/data/heatmaps")
MODEL_PATH = Path("refinement/checkpoints/refinement_net.pt")


def main():
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print("ERROR: Run 'python -m refinement.generate_heatmaps' first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    valid = [r for r in all_results if "distance_px" in r]
    print(f"Total valid samples: {len(valid)}")

    # Baseline: raw attention centroid
    baseline_dists = [r["distance_px"] for r in valid]
    print(f"\n--- BASELINE (raw attention centroid) ---")
    print(f"Mean distance: {np.mean(baseline_dists):.1f}px")
    print(f"Median distance: {np.median(baseline_dists):.1f}px")
    print(f"Pass (<50px): {sum(1 for d in baseline_dists if d < 50)}/{len(valid)} ({100*sum(1 for d in baseline_dists if d < 50)/len(valid):.1f}%)")
    print(f"Pass (<30px): {sum(1 for d in baseline_dists if d < 30)}/{len(valid)} ({100*sum(1 for d in baseline_dists if d < 30)/len(valid):.1f}%)")

    # Refinement net
    if not MODEL_PATH.exists():
        print(f"\nNo refinement model found at {MODEL_PATH}. Run 'python -m refinement.train' first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    heatmap_rows = checkpoint["heatmap_rows"]
    heatmap_cols = checkpoint["heatmap_cols"]
    model = RefinementNet(heatmap_rows=heatmap_rows, heatmap_cols=heatmap_cols)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    refined_dists = []
    for r in valid:
        heatmap = np.load(OUTPUT_DIR / r["heatmap_file"]).astype(np.float32)
        heatmap_2d = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, rows, cols)

        w, h = r["image_size"]
        centroid_norm = torch.tensor([[
            r["predicted_click"][0] / w,
            r["predicted_click"][1] / h,
        ]], dtype=torch.float32)
        argmax_norm = torch.tensor([[
            r["argmax_click"][0] / w,
            r["argmax_click"][1] / h,
        ]], dtype=torch.float32)

        with torch.no_grad():
            pred = model(heatmap_2d, centroid_norm, argmax_norm)

        pred_x = pred[0, 0].item() * w
        pred_y = pred[0, 1].item() * h
        true_x, true_y = r["true_click"]

        dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        refined_dists.append(dist)

    print(f"\n--- REFINEMENT CNN ---")
    print(f"Mean distance: {np.mean(refined_dists):.1f}px")
    print(f"Median distance: {np.median(refined_dists):.1f}px")
    print(f"Pass (<50px): {sum(1 for d in refined_dists if d < 50)}/{len(valid)} ({100*sum(1 for d in refined_dists if d < 50)/len(valid):.1f}%)")
    print(f"Pass (<30px): {sum(1 for d in refined_dists if d < 30)}/{len(valid)} ({100*sum(1 for d in refined_dists if d < 30)/len(valid):.1f}%)")

    # Improvement
    print(f"\n--- IMPROVEMENT ---")
    print(f"Mean: {np.mean(baseline_dists):.1f}px -> {np.mean(refined_dists):.1f}px ({np.mean(baseline_dists) - np.mean(refined_dists):+.1f}px)")
    print(f"Median: {np.median(baseline_dists):.1f}px -> {np.median(refined_dists):.1f}px ({np.median(baseline_dists) - np.median(refined_dists):+.1f}px)")

    # Per-source breakdown
    print(f"\n--- BY SOURCE ---")
    sources = set(r["data_source"] for r in valid)
    for src in sorted(sources):
        src_baseline = [r["distance_px"] for r in valid if r["data_source"] == src]
        src_indices = [i for i, r in enumerate(valid) if r["data_source"] == src]
        src_refined = [refined_dists[i] for i in src_indices]
        print(f"  {src:<10} baseline={np.mean(src_baseline):.1f}px  refined={np.mean(src_refined):.1f}px  n={len(src_baseline)}")


if __name__ == "__main__":
    main()
