from huggingface_hub import snapshot_download
import os

# Get the current working directory
current_dir = os.getcwd()

# Download the dataset repo here
local_dir = snapshot_download(
    repo_id="hsanth01/ragx-databases",
    repo_type="dataset",
    local_dir=current_dir,
    local_dir_use_symlinks=False  # ensures full copies instead of symlinks
)

print(f"Files downloaded to: {local_dir}")
