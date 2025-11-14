#!/usr/bin/env python3
import os
import tarfile
from tqdm import tqdm

# Paths
tar_root = "/home/ubuntu/datasets/imagenet-1k-wds"  # location of val tar files
dst_root = "/home/ubuntu/datasets/imagenet/val"     # output folder

os.makedirs(dst_root, exist_ok=True)

# List all validation tar files
tars = sorted([f for f in os.listdir(tar_root) 
               if f.startswith("imagenet1k-val") and f.endswith(".tar")])

print(f"Found {len(tars)} val tar files.")  # Debug line

for tar_name in tqdm(tars, desc="Processing val tar files"):
    tar_path = os.path.join(tar_root, tar_name)
    
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            f = tar.extractfile(member)
            filename = os.path.basename(member.name)

            # The synset is the prefix before the first underscore
            synset = filename.split("_")[0]

            # Make folder for the synset
            class_dir = os.path.join(dst_root, synset)
            os.makedirs(class_dir, exist_ok=True)

            # Save the JPEG
            out_path = os.path.join(class_dir, filename)
            with open(out_path, "wb") as out_f:
                out_f.write(f.read())

    # Delete the tar to free space
    os.remove(tar_path)
    tqdm.write(f"✅ Extracted and removed: {tar_name}")

print("🎉 All validation shards processed.")
