# #!/usr/bin/env python3
# import os
# import tarfile
# from tqdm import tqdm

# tar_root = "/home/ubuntu/datasets/imagenet-1k-wds"
# dst_root = "/home/ubuntu/datasets/imagenet/train"

# os.makedirs(dst_root, exist_ok=True)

# # List all train tar files
# tars = sorted([f for f in os.listdir(tar_root) if f.startswith("imagenet1k-train") and f.endswith(".tar")])

# for tar_name in tqdm(tars, desc="Processing train tar files"):
#     tar_path = os.path.join(tar_root, tar_name)
    
#     with tarfile.open(tar_path, "r") as tar:
#         for member in tar.getmembers():
#             if not member.isfile():
#                 continue

#             f = tar.extractfile(member)
#             filename = os.path.basename(member.name)

#             # The synset is the prefix before the first underscore
#             synset = filename.split("_")[0]

#             # Make folder for the synset
#             class_dir = os.path.join(dst_root, synset)
#             os.makedirs(class_dir, exist_ok=True)

#             # Save the JPEG
#             out_path = os.path.join(class_dir, filename)
#             with open(out_path, "wb") as out_f:
#                 out_f.write(f.read())

#     # Delete tar to save space
#     os.remove(tar_path)
#     tqdm.write(f"✅ Extracted and removed: {tar_name}")

# print("🎉 All train shards processed.")
#!/usr/bin/env python3
import os
import tarfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Paths
tar_root = "/home/ubuntu/datasets/imagenet-1k-wds"
dst_root = "/home/ubuntu/datasets/imagenet/train"
os.makedirs(dst_root, exist_ok=True)

# List all train tar files
all_tars = sorted([f for f in os.listdir(tar_root) 
                   if f.startswith("imagenet1k-train") and f.endswith(".tar")])

print(f"Found {len(all_tars)} train tar files.")

# Function to extract a single tar file
def extract_tar(tar_name):
    tar_path = os.path.join(tar_root, tar_name)
    
    # Check if already extracted (resume support)
    # We assume that if the first file exists, this tar is done
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            return f"Empty tar skipped: {tar_name}"
        first_file = os.path.basename(members[0].name)
        synset = first_file.split("_")[0]
        class_dir = os.path.join(dst_root, synset)
        os.makedirs(class_dir, exist_ok=True)
        out_path = os.path.join(class_dir, first_file)
        if os.path.exists(out_path):
            return f"Already extracted, skipping: {tar_name}"

    # Extract tar
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            filename = os.path.basename(member.name)
            synset = filename.split("_")[0]
            class_dir = os.path.join(dst_root, synset)
            os.makedirs(class_dir, exist_ok=True)
            out_path = os.path.join(class_dir, filename)
            with open(out_path, "wb") as out_f:
                out_f.write(f.read())
    
    # Delete tar to save space
    os.remove(tar_path)
    return f"✅ Extracted and removed: {tar_name}"

# Use multiprocessing pool
num_workers = min(cpu_count(), 8)  # adjust max parallel processes
with Pool(num_workers) as pool:
    for result in tqdm(pool.imap_unordered(extract_tar, all_tars), total=len(all_tars)):
        tqdm.write(result)

print("🎉 All train shards processed (parallel & resumable).")
