import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import append_to_manifest, load_manifest, generate_image_id

def load_yolo_acne():
    """Load acne-dataset-in-yolov8-format."""
    source_dir = PROJECT_ROOT / "data" / "raw" / "acne-dataset-in-yolov8-format" / "data-2"
    
    rows = []
    if not source_dir.exists():
        return
        
    for split in ["train", "valid", "test"]:
        split_dir = source_dir / split / "images"
        if not split_dir.exists():
            continue
            
        for img_path in split_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            filename = img_path.name
            image_id = generate_image_id("yolo_acne", filename)
            rel_to_root = str(img_path.relative_to(PROJECT_ROOT).as_posix())
            
            rows.append({
                "image_id": image_id,
                "source": "yolo_acne",
                "subject_id": None,
                "path": rel_to_root,
                "orig_label": "Acne",
                "label": "Acne",
                "role": "train_pool"
            })
            
    if rows:
        df = pd.DataFrame(rows)
        manifest_df = load_manifest()
        new_ids = df[~df["image_id"].isin(manifest_df["image_id"])]
        if not new_ids.empty:
            append_to_manifest(new_ids)

if __name__ == "__main__":
    load_yolo_acne()
