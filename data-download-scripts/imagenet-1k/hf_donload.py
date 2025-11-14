#!/usr/bin/env python3
from huggingface_hub import snapshot_download, login

# ✅ Replace with your real token
HF_TOKEN = "NONE"

login(token=HF_TOKEN)

local_dir = "/home/ubuntu/datasets/imagenet-1k-wds"

snapshot_download(
    repo_id="timm/imagenet-1k-wds",
    repo_type="dataset",
    local_dir=local_dir,     # will resume if interrupted automatically
)

print("✅ Download complete (or resumed). Files are in:", local_dir)
