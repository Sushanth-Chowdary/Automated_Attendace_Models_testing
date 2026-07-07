import os
import shutil
import kagglehub

# Your specific network share path
target_dir = "/run/user/1000/gvfs/smb-share:server=10.23.20.56,share=rkgnas_user2/EE23B044_testing/lfw-dataset"

print("Downloading dataset to local cache...")
# Download latest version to cache
cached_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print("Cached path:", cached_path)

print(f"Copying files to {target_dir}...")
# Copy from the cache to your specific folder
shutil.copytree(cached_path, target_dir, dirs_exist_ok=True)

print("Dataset successfully moved to your target folder!")