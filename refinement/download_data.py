"""
Download ScreenSpot dataset from HuggingFace.

Saves images and annotations to refinement/data/screenspot/

Usage:
    python -m refinement.download_data
"""

from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("refinement/data/screenspot")


def main():
    print("Downloading ScreenSpot dataset...")
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")

    print(f"Dataset size: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"\nFirst sample:")
    sample = ds[0]
    print(f"  instruction: {sample['instruction']}")
    print(f"  bbox: {sample['bbox']}")
    print(f"  data_type: {sample['data_type']}")
    print(f"  data_source: {sample['data_source']}")
    print(f"  image size: {sample['image'].size}")

    # Save images and metadata
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    import json
    annotations = []

    for i, sample in enumerate(ds):
        img = sample["image"]
        img_path = images_dir / f"{i:04d}.png"
        img.save(img_path)

        bbox = sample["bbox"]  # normalized [x1, y1, x2, y2]
        w, h = img.size
        # Compute click point (bbox center) in absolute pixels
        cx = ((bbox[0] + bbox[2]) / 2) * w
        cy = ((bbox[1] + bbox[3]) / 2) * h

        annotations.append({
            "id": i,
            "image": f"{i:04d}.png",
            "instruction": sample["instruction"],
            "bbox_norm": bbox,
            "click_point": [cx, cy],
            "image_size": [w, h],
            "data_type": sample["data_type"],
            "data_source": sample["data_source"],
        })

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(ds)} images...")

    annotations_path = DATA_DIR / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone. Saved {len(annotations)} samples to {DATA_DIR}")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_path}")

    # Print stats
    sources = {}
    types = {}
    for a in annotations:
        sources[a["data_source"]] = sources.get(a["data_source"], 0) + 1
        types[a["data_type"]] = types.get(a["data_type"], 0) + 1

    print(f"\nBy source: {sources}")
    print(f"By type: {types}")


if __name__ == "__main__":
    main()
