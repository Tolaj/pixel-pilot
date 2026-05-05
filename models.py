import os
from huggingface_hub import hf_hub_download


def download_if_missing(repo_id, files, local_dir):
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            print(f"Already exists, skipping: {local_path}")
            continue
        print(f"Downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"Done: {local_path}")


# Qwen3-VL-2B-Instruct
download_if_missing(
    repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
    local_dir="./models/Qwen3-VL-2B-Instruct-GGUF",
    files=[
        "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
    ],
)

# OmniParser
OMNI_DIR = "./models/omniparser"
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

# Rename icon_caption → icon_caption_florence if not done yet
src = os.path.join(OMNI_DIR, "icon_caption")
dst = os.path.join(OMNI_DIR, "icon_caption_florence")
if os.path.exists(src) and not os.path.exists(dst):
    os.rename(src, dst)
    print("Renamed icon_caption → icon_caption_florence")
elif os.path.exists(dst):
    print("icon_caption_florence already exists, skipping rename")
