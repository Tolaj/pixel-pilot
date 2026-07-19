"""
Train the refinement CNN on generated heatmaps + ScreenSpot ground truth.

Usage:
    python -m refinement.train
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from refinement.model import RefinementNet

OUTPUT_DIR = Path("refinement/data/heatmaps")
MODEL_SAVE_PATH = Path("refinement/checkpoints/refinement_net.pt")

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
VAL_SPLIT = 0.2


class HeatmapDataset(Dataset):
    """Dataset of (heatmap_2d, predicted_point, true_point) tuples."""

    def __init__(self, results_path: Path, heatmaps_dir: Path):
        with open(results_path) as f:
            all_results = json.load(f)

        self.samples = [r for r in all_results if "distance_px" in r]
        self.heatmaps_dir = heatmaps_dir
        print(f"Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]

        # Load heatmap as 2D and add channel dim -> (1, rows, cols)
        heatmap = np.load(self.heatmaps_dir / r["heatmap_file"]).astype(np.float32)
        heatmap_2d = torch.from_numpy(heatmap).unsqueeze(0)

        w, h = r["image_size"]
        centroid_norm = torch.tensor([
            r["predicted_click"][0] / w,
            r["predicted_click"][1] / h,
        ], dtype=torch.float32)

        argmax_norm = torch.tensor([
            r["argmax_click"][0] / w,
            r["argmax_click"][1] / h,
        ], dtype=torch.float32)

        true_norm = torch.tensor([
            r["true_click"][0] / w,
            r["true_click"][1] / h,
        ], dtype=torch.float32)

        return heatmap_2d, centroid_norm, argmax_norm, true_norm


def train():
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print("ERROR: Run 'python -m refinement.generate_heatmaps' first.")
        return

    dataset = HeatmapDataset(results_path, OUTPUT_DIR)

    if len(dataset) < 10:
        print(f"ERROR: Only {len(dataset)} samples. Need more data.")
        return

    # Determine heatmap shape from first sample
    sample = dataset[0]
    _, heatmap_rows, heatmap_cols = sample[0].shape
    print(f"Heatmap shape: {heatmap_rows}x{heatmap_cols}")

    # Split train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Train: {train_size}, Val: {val_size}")

    model = RefinementNet(heatmap_rows=heatmap_rows, heatmap_cols=heatmap_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    loss_fn = nn.MSELoss()

    with open(results_path) as f:
        all_results = json.load(f)
    valid_results = [r for r in all_results if "distance_px" in r]

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for heatmap, centroid, argmax, true in train_loader:
            pred = model(heatmap, centroid, argmax)
            loss = loss_fn(pred, true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * heatmap.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for heatmap, centroid, argmax, true in val_loader:
                pred = model(heatmap, centroid, argmax)
                loss = loss_fn(pred, true)
                val_loss += loss.item() * heatmap.size(0)
                val_preds.append(pred)
                val_trues.append(true)
        val_loss /= val_size

        scheduler.step(val_loss)

        if val_preds:
            preds = torch.cat(val_preds).numpy()
            trues = torch.cat(val_trues).numpy()
            avg_w = np.mean([r["image_size"][0] for r in valid_results])
            avg_h = np.mean([r["image_size"][1] for r in valid_results])
            pixel_dists = np.sqrt(
                ((preds[:, 0] - trues[:, 0]) * avg_w)**2 +
                ((preds[:, 1] - trues[:, 1]) * avg_h)**2
            )
            mean_px = np.mean(pixel_dists)
        else:
            mean_px = 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mean_px={mean_px:.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "heatmap_rows": heatmap_rows,
                "heatmap_cols": heatmap_cols,
                "val_loss": val_loss,
                "epoch": epoch,
            }, MODEL_SAVE_PATH)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
    print(f"Model saved: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
