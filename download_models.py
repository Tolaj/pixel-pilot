from huggingface_hub import hf_hub_download, snapshot_download
import os

# Download full Florence-2 base (processor + config + weights)
print("Downloading Florence-2 base...")
snapshot_download(
    repo_id="microsoft/Florence-2-base-ft",
    local_dir="./models/omniparser/icon_caption_florence",
)

# Override weights with OmniParser v2 fine-tuned model
print("Downloading OmniParser v2 fine-tuned caption weights...")
hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_caption/model.safetensors",
    local_dir="./models/omniparser",
)

# Replace the weights
import shutil

src = "./models/omniparser/icon_caption/model.safetensors"
dst = "./models/omniparser/icon_caption_florence/model.safetensors"
shutil.copy(src, dst)
print(f"Replaced weights: {dst}")
