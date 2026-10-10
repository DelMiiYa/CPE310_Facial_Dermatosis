import os
import sys
import logging
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.sources.scin import load_cases, explode_images, CLASS_GROUPS, group_weight
from src.manifest import append_to_manifest, load_manifest, generate_image_id

SCIN_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "scin" / "dataset" / "images"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def download_image(url: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return True
    
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        part_path.rename(dest_path)
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        if part_path.exists():
            part_path.unlink()
        return False

def get_scin_filtered() -> pd.DataFrame:
    cases = load_cases()
    
    # 1. Head or neck only
    cases = cases[cases["head_or_neck"] == True].copy()
    
    # Define our target groups
    target_groups = {
        "Acne": CLASS_GROUPS["Acne"]["broad"],
        "Rosacea": CLASS_GROUPS["Rosacea"]["broad"],
        "Dermatitis": CLASS_GROUPS["Eczema"]["broad"] + CLASS_GROUPS["Contact dermatitis"]["broad"]
    }
    
    rows = []
    for _, case in cases.iterrows():
        weights = case["weighted_label"]
        if not weights:
            continue
            
        qualifying_classes = []
        for class_name, synonyms in target_groups.items():
            w = group_weight(weights, synonyms)
            if w >= 0.5:
                qualifying_classes.append(class_name)
        
        if len(qualifying_classes) == 1:
            case_dict = case.to_dict()
            case_dict["assigned_label"] = qualifying_classes[0]
            case_dict["assigned_confidence"] = group_weight(weights, target_groups[qualifying_classes[0]])
            rows.append(case_dict)
        elif len(qualifying_classes) > 1:
            logging.warning(f"Case {case['case_id']} qualifies for multiple classes: {qualifying_classes}. Skipping.")
            
    return pd.DataFrame(rows)

def main():
    SCIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    filtered_cases = get_scin_filtered()
    logging.info(f"Found {len(filtered_cases)} qualifying facial cases.")
    
    images_df = explode_images(filtered_cases)
    # Merge assigned label back
    images_df = images_df.merge(filtered_cases[["case_id", "assigned_label", "assigned_confidence", "efst", "dermatologist_skin_condition_on_label_name"]], on="case_id", how="left")
    
    logging.info(f"Total images to download: {len(images_df)}")
    
    # Check what's already in manifest
    manifest_df = load_manifest()
    existing_paths = set(manifest_df["path"].dropna().tolist())
    
    manifest_rows = []
    download_tasks = []
    
    for _, row in images_df.iterrows():
        case_id = row["case_id"]
        rel_path = row["image_path"] # e.g., dataset/images/abc.png
        filename = Path(rel_path).name
        url = f"https://storage.googleapis.com/dx-scin-public-data/{rel_path}"
        
        dest_path = SCIN_IMAGES_DIR / filename
        rel_to_root = str(dest_path.relative_to(PROJECT_ROOT).as_posix())
        
        image_id = generate_image_id("scin", filename)
        
        if image_id not in manifest_df["image_id"].values:
            manifest_rows.append({
                "image_id": image_id,
                "source": "scin",
                "subject_id": case_id,
                "path": rel_to_root,
                "orig_label": row["dermatologist_skin_condition_on_label_name"],
                "label": row["assigned_label"],
                "label_confidence": row["assigned_confidence"],
                "body_part": "head_or_neck",
                "shot_type": row["shot_type"],
                "fst": row["efst"],
                "role": "external_test",
                "review_status": "accepted", # Automatically accepted for test set
            })
        
        download_tasks.append((url, dest_path))

    # Download in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_image, url, dest): dest for url, dest in download_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if future.result():
                success_count += 1
                
    logging.info(f"Successfully downloaded/verified {success_count}/{len(download_tasks)} images.")
    
    if manifest_rows:
        new_df = pd.DataFrame(manifest_rows)
        # Drop any duplicate image IDs from the new rows
        new_df = new_df.drop_duplicates(subset=["image_id"], keep="first")
        append_to_manifest(new_df)
    else:
        logging.info("No new rows to append to manifest.")

if __name__ == "__main__":
    main()
