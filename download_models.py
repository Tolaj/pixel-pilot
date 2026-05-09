# models.py
# ── download flags ─────────────────────────────────────────────────────────────
# Set to True/False to control what gets downloaded

DOWNLOAD_QWEN3_VL = False  # Qwen3-VL-2B multimodal (for vision agent)
DOWNLOAD_OMNIPARSER = False  # OmniParser v2 (YOLO icon detector)
DOWNLOAD_FLORENCE2 = False  # Florence-2 base + OmniParser caption weights
DOWNLOAD_MOONDREAM2 = True  # Moondream2 (fast screenshot-to-action VLM)

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


def snapshot_if_missing(repo_id, local_dir, ignore_patterns=None):
    # consider present if dir exists and has at least one file
    if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        print(f"  already exists, skipping: {local_dir}")
        return
    print(f"  downloading snapshot → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=ignore_patterns or [],
    )
    print(f"  done: {local_dir}")


# ── Qwen3-VL-2B (multimodal GGUF) ────────────────────────────────────────────
if DOWNLOAD_QWEN3_VL:
    print("\n[1/4] Qwen3-VL-2B-Instruct GGUF")
    download_if_missing(
        repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
        local_dir="./models/Qwen3-VL-2B-Instruct-GGUF",
        files=[
            "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
        ],
    )
else:
    print("\n[1/4] Qwen3-VL — skipped (DOWNLOAD_QWEN3_VL=False)")


# ── OmniParser v2 (YOLO icon detector) ───────────────────────────────────────
OMNI_DIR = "./models/omniparser"

if DOWNLOAD_OMNIPARSER:
    print("\n[2/4] OmniParser v2 — icon detector")
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
    print("\n[2/4] OmniParser — skipped (DOWNLOAD_OMNIPARSER=False)")


# ── Florence-2 base + OmniParser caption weights ──────────────────────────────
if DOWNLOAD_FLORENCE2:
    print("\n[3/4] Florence-2 base + OmniParser caption weights")

    FLORENCE_DIR = os.path.join(OMNI_DIR, "icon_caption_florence")

    # full Florence-2 base (processor, config, tokenizer, weights)
    snapshot_if_missing(
        repo_id="microsoft/Florence-2-base-ft",
        local_dir=FLORENCE_DIR,
    )

    # override weights with OmniParser v2 fine-tuned checkpoint
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

    # tidy up leftover icon_caption dir if it came from models.py rename logic
    legacy_src = os.path.join(OMNI_DIR, "icon_caption")
    legacy_dst = os.path.join(OMNI_DIR, "icon_caption_florence")
    if os.path.exists(legacy_src) and not os.path.exists(legacy_dst):
        os.rename(legacy_src, legacy_dst)
        print("  renamed icon_caption → icon_caption_florence")
    elif os.path.exists(legacy_dst):
        print("  icon_caption_florence already exists, skipping rename")

else:
    print("\n[3/4] Florence-2 — skipped (DOWNLOAD_FLORENCE2=False)")


# ── Moondream2 ────────────────────────────────────────────────────────────────
# Moondream2 is a 1.8B VLM optimized for fast screenshot understanding.
# Much faster than Qwen3-VL for GUI tasks; pure-Python, no llama-cpp server needed.
# Usage after download:
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   model = AutoModelForCausalLM.from_pretrained("./models/moondream2", trust_remote_code=True)
#   tokenizer = AutoTokenizer.from_pretrained("./models/moondream2")

if DOWNLOAD_MOONDREAM2:
    print("\n[4/4] Moondream2 (1.8B fast VLM)")
    snapshot_if_missing(
        repo_id="vikhyatk/moondream2",
        local_dir="./models/moondream2",
        # skip git history and large unused checkpoints
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
else:
    print("\n[4/4] Moondream2 — skipped (DOWNLOAD_MOONDREAM2=False)")


print("\nAll selected downloads complete.")
