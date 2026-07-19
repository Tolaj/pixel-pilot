"""
Download model weights from Hugging Face.

Usage:
    python scripts/download_models.py

Toggle flags below to control what gets downloaded.
"""

# ── download flags ────────────────────────────────────────────────────────────
DOWNLOAD_QWEN2_VL = False
DOWNLOAD_QWEN3_VL = True
DOWNLOAD_OMNIPARSER = False
DOWNLOAD_FLORENCE2 = False
DOWNLOAD_MOONDREAM2 = False

# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil
from huggingface_hub import hf_hub_download, snapshot_download


def download_if_missing(repo_id, files, local_dir):
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            print(f"  already exists, skipping: {local_path}")
            continue
        print(f"  downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"  done: {local_path}")


def snapshot_if_missing(repo_id, local_dir, ignore_patterns=None, required_suffix=".safetensors"):
    if os.path.isdir(local_dir):
        has_weights = any(
            f.name.endswith(required_suffix)
            for f in os.scandir(local_dir)
            if f.is_file()
        )
        if has_weights:
            print(f"  already exists, skipping: {local_dir}")
            return
    print(f"  downloading snapshot -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=ignore_patterns or [],
    )
    print(f"  done: {local_dir}")


def main():
    if DOWNLOAD_QWEN2_VL:
        print("\n[1/5] Qwen2-VL-2B-Instruct")
        snapshot_if_missing(
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            local_dir="./models/Qwen2-VL-2B-Instruct",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
    else:
        print("\n[1/5] Qwen2-VL-2B -- skipped")

    if DOWNLOAD_QWEN3_VL:
        print("\n[2/5] Qwen3-VL-2B-Instruct")
        snapshot_if_missing(
            repo_id="Qwen/Qwen3-VL-2B-Instruct",
            local_dir="./models/Qwen3-VL-2B-Instruct",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
    else:
        print("\n[2/5] Qwen3-VL-2B -- skipped")

    OMNI_DIR = "./models/omniparser"

    if DOWNLOAD_OMNIPARSER:
        print("\n[3/5] OmniParser v2")
        download_if_missing(
            repo_id="microsoft/OmniParser-v2.0",
            local_dir=OMNI_DIR,
            files=[
                "icon_detect/train_args.yaml",
                "icon_detect/model.pt",
                "icon_detect/model.yaml",
                "icon_caption/config.json",
                "icon_caption/generation_config.json",
                "icon_caption/model.safetensors",
            ],
        )
    else:
        print("\n[3/5] OmniParser -- skipped")

    if DOWNLOAD_FLORENCE2:
        print("\n[4/5] Florence-2 base + OmniParser caption weights")
        FLORENCE_DIR = os.path.join(OMNI_DIR, "icon_caption_florence")
        snapshot_if_missing(
            repo_id="microsoft/Florence-2-base-ft",
            local_dir=FLORENCE_DIR,
        )
        print("  downloading OmniParser v2 caption weights...")
        hf_hub_download(
            repo_id="microsoft/OmniParser-v2.0",
            filename="icon_caption/model.safetensors",
            local_dir=OMNI_DIR,
        )
        src = os.path.join(OMNI_DIR, "icon_caption", "model.safetensors")
        dst = os.path.join(FLORENCE_DIR, "model.safetensors")
        shutil.copy(src, dst)
        print(f"  replaced weights: {dst}")
    else:
        print("\n[4/5] Florence-2 -- skipped")

    if DOWNLOAD_MOONDREAM2:
        print("\n[5/5] Moondream2")
        snapshot_if_missing(
            repo_id="vikhyatk/moondream2",
            local_dir="./models/moondream2",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
    else:
        print("\n[5/5] Moondream2 -- skipped")

    print("\nAll selected downloads complete.")


if __name__ == "__main__":
    main()
