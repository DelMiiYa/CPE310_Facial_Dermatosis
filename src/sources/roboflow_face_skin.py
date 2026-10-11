import logging
from pathlib import Path
import pandas as pd

from src.manifest import PROJECT_ROOT, append_to_manifest, load_manifest, generate_image_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LABEL_MAPPING = {
    "rosacea": "Rosacea",
    "eksim": "Dermatitis",
    "acne": "Acne",
}

def load_roboflow_face_skin():
    """Load downloaded Roboflow face-skin-diseases dataset into manifest.
    
    Maps rosacea -> Rosacea, eksim -> Dermatitis, acne -> Acne.
    Skips out-of-scope classes (herpes, panu).
    """
    source_dir = PROJECT_ROOT / "data" / "raw" / "roboflow-face-skin"
    if not source_dir.exists():
        logging.warning(f"Roboflow face skin directory does not exist: {source_dir}")
        return

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []

    # Dataset structure in folder format: split/class/filename
    for split in ["train", "test"]:
        split_dir = source_dir / split
        if not split_dir.exists():
            continue

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue

            orig_label = cls_dir.name.lower()
            if orig_label not in LABEL_MAPPING:
                # Skip herpes, panu, or other out-of-scope classes
                continue

            target_label = LABEL_MAPPING[orig_label]

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in img_exts:
                    continue

                filename = img_path.name
                rel_to_root = str(img_path.relative_to(PROJECT_ROOT).as_posix())
                image_id = generate_image_id("roboflow_face_skin", filename)

                rows.append({
                    "image_id": image_id,
                    "source": "roboflow_face_skin",
                    "subject_id": None,
                    "path": rel_to_root,
                    "orig_label": orig_label,
                    "label": target_label,
                    "role": "train_pool",
                    "review_status": "pending",
                })

    if rows:
        df = pd.DataFrame(rows)
        manifest_df = load_manifest()
        new_ids = df[~df["image_id"].isin(manifest_df["image_id"])]
        if not new_ids.empty:
            append_to_manifest(new_ids)
            logging.info(f"Appended {len(new_ids)} images from roboflow_face_skin to manifest.")
        else:
            logging.info("All roboflow_face_skin images are already present in manifest.")
    else:
        logging.warning("No images loaded from roboflow_face_skin.")

if __name__ == "__main__":
    load_roboflow_face_skin()
