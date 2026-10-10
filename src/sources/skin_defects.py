import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import append_to_manifest, load_manifest, generate_image_id

def load_skin_defects():
    """Load skin-defects-acne-redness-and-bags-under-the-eyes."""
    source_dir = PROJECT_ROOT / "data" / "raw" / "skin-defects-acne-redness-and-bags-under-the-eyes" / "files" / "acne"
    
    rows = []
    if not source_dir.exists():
        return
        
    for person_dir in source_dir.iterdir():
        if not person_dir.is_dir():
            continue
            
        subject_id = person_dir.name
        
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            filename = img_path.name
            image_id = generate_image_id("skin_defects", f"{subject_id}_{filename}")
            rel_to_root = str(img_path.relative_to(PROJECT_ROOT).as_posix())
            
            rows.append({
                "image_id": image_id,
                "source": "skin_defects",
                "subject_id": subject_id,
                "path": rel_to_root,
                "orig_label": "acne",
                "label": "Acne",
                "shot_type": filename.replace(img_path.suffix, ""), # front, left_side, right_side
                "role": "train_pool"
            })
            
    if rows:
        df = pd.DataFrame(rows)
        manifest_df = load_manifest()
        new_ids = df[~df["image_id"].isin(manifest_df["image_id"])]
        if not new_ids.empty:
            append_to_manifest(new_ids)

if __name__ == "__main__":
    load_skin_defects()
