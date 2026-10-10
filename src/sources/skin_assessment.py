import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import append_to_manifest, load_manifest, generate_image_id

def load_skin_assessment():
    """Load skin-assessment-dataset."""
    source_dir = PROJECT_ROOT / "data" / "raw" / "skin-assessment-dataset" / "skin_assesment_dataset"
    
    label_map = {
        "acne": "Acne",
        "Rosacea": "Rosacea",
        "Eczema": "Dermatitis",
        "normal": "Normal"
    }
    
    rows = []
    if not source_dir.exists():
        return
        
    for class_folder, target_label in label_map.items():
        class_dir = source_dir / class_folder
        if not class_dir.exists():
            continue
            
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            filename = img_path.name
            image_id = generate_image_id("skin_assessment", f"{class_folder}_{filename}")
            rel_to_root = str(img_path.relative_to(PROJECT_ROOT).as_posix())
            
            rows.append({
                "image_id": image_id,
                "source": "skin_assessment",
                "subject_id": None,
                "path": rel_to_root,
                "orig_label": class_folder,
                "label": target_label,
                "role": "train_pool"
            })
            
    if rows:
        df = pd.DataFrame(rows)
        manifest_df = load_manifest()
        new_ids = df[~df["image_id"].isin(manifest_df["image_id"])]
        if not new_ids.empty:
            append_to_manifest(new_ids)

if __name__ == "__main__":
    load_skin_assessment()
