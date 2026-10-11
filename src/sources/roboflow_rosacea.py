import json
import logging
from pathlib import Path
import pandas as pd

from src.manifest import PROJECT_ROOT, append_to_manifest, load_manifest, generate_image_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_roboflow_rosacea():
    """Load downloaded Roboflow rosacea dataset into manifest."""
    source_dir = PROJECT_ROOT / "data" / "raw" / "roboflow-rosacea"
    if not source_dir.exists():
        logging.warning(f"Roboflow rosacea directory does not exist: {source_dir}")
        return

    splits = ["train", "valid", "test"]
    rows = []

    for split in splits:
        split_dir = source_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        if not split_dir.exists() or not ann_file.exists():
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Map category_id to category name
        cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

        # Find which images have 'redness' / rosacea annotations
        annotated_images = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_name = cat_map.get(ann.get("category_id"))
            if cat_name == "redness":
                annotated_images[img_id] = "redness"

        for img_info in coco_data.get("images", []):
            img_id = img_info["id"]
            if img_id not in annotated_images:
                # Exclude unannotated images or non-rosacea images
                continue

            filename = img_info["file_name"]
            img_path = split_dir / filename
            if not img_path.exists():
                continue

            rel_to_root = str(img_path.relative_to(PROJECT_ROOT).as_posix())
            image_id = generate_image_id("roboflow_rosacea", filename)

            rows.append({
                "image_id": image_id,
                "source": "roboflow_rosacea",
                "subject_id": None,
                "path": rel_to_root,
                "orig_label": "redness",
                "label": "Rosacea",
                "role": "train_pool",
                "review_status": "pending"
            })

    if rows:
        df = pd.DataFrame(rows)
        manifest_df = load_manifest()
        new_ids = df[~df["image_id"].isin(manifest_df["image_id"])]
        if not new_ids.empty:
            append_to_manifest(new_ids)
            logging.info(f"Appended {len(new_ids)} images from roboflow_rosacea to manifest.")
        else:
            logging.info("All roboflow_rosacea images are already present in manifest.")
    else:
        logging.warning("No images loaded from roboflow_rosacea.")

if __name__ == "__main__":
    load_roboflow_rosacea()
