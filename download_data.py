import os
import subprocess
import sys

kaggle_datasets = [
    "jhirshaikh/skin-assessment-dataset",
    "trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes",
    "osmankagankurnaz/facial-skin-diseases-dataset",
    "osmankagankurnaz/acne-dataset-in-yolov8-format",
    "muhammadabdulsami/massive-skin-disease-balanced-dataset",
    "tapakah68/skin-problems-34-on-the-iga-scale"
]

base_dir = "data/raw"
os.makedirs(base_dir, exist_ok=True)

for ds in kaggle_datasets:
    ds_name = ds.split("/")[-1]
    target_dir = os.path.join(base_dir, ds_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {ds} into {target_dir}...")
    
    cmd = f"kaggle datasets download -d {ds} -p {target_dir} --unzip"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error downloading {ds}")

print("Cloning SCIN dataset from GitHub...")
if not os.path.exists(os.path.join(base_dir, "scin")):
    subprocess.run(f"git clone https://github.com/google-research-datasets/scin.git {os.path.join(base_dir, 'scin')}", shell=True)
else:
    print("SCIN dataset directory already exists. Skipping clone.")

print("Note: The Roboflow dataset requires an API key to download automatically. We will review the Kaggle and SCIN datasets first.")
print("Downloads script completed.")
