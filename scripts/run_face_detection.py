import argparse
import logging
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import load_manifest, MANIFEST_PATH, COLUMNS
from src.detection import FaceDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_detection(
    manifest_path: Path = MANIFEST_PATH,
    force: bool = False,
    save_every: int = 200,
    max_dim: int = 1280,
):
    # Ensure backup exists
    backup_path = manifest_path.parent / "manifest_before_step5.csv"
    if not backup_path.exists():
        logging.info(f"Creating backup of manifest at {backup_path}")
        shutil.copy2(manifest_path, backup_path)
    else:
        logging.info(f"Backup already exists at {backup_path}")

    df = load_manifest()
    total_rows = len(df)
    logging.info(f"Loaded manifest with {total_rows} rows.")

    # Validate file existence
    missing_files = []
    for idx, row in df.iterrows():
        p = PROJECT_ROOT / row["path"]
        if not p.exists():
            missing_files.append((idx, str(p)))
    if missing_files:
        logging.warning(f"Found {len(missing_files)} missing files! First few: {missing_files[:5]}")
    else:
        logging.info("All manifest image files verified to exist on disk.")

    # Determine rows to process
    if force:
        pending_mask = pd.Series(True, index=df.index)
    else:
        pending_mask = df["face_detected"].isna()

    pending_indices = df[pending_mask].index.tolist()
    logging.info(f"Rows to process: {len(pending_indices)} (out of {total_rows})")

    if not pending_indices:
        logging.info("All rows already have face detection results.")
        return df

    failures = []
    processed_count = 0

    with FaceDetector() as detector:
        pbar = tqdm(pending_indices, desc="Face detection", unit="img")
        for idx in pbar:
            row = df.loc[idx]
            img_path = PROJECT_ROOT / row["path"]
            
            img = cv2.imread(str(img_path))
            if img is None:
                failures.append((idx, str(img_path)))
                df.at[idx, "face_detected"] = False
                df.at[idx, "face_conf"] = np.nan
                df.at[idx, "face_area_frac"] = np.nan
            else:
                try:
                    res = detector.detect(img, max_dim=max_dim)
                    df.at[idx, "face_detected"] = bool(res["face_detected"])
                    df.at[idx, "face_conf"] = float(res["face_conf"]) if res["face_conf"] is not None else np.nan
                    df.at[idx, "face_area_frac"] = float(res["face_area_frac"]) if res["face_area_frac"] is not None else np.nan
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
                    failures.append((idx, str(img_path)))
                    df.at[idx, "face_detected"] = False
                    df.at[idx, "face_conf"] = np.nan
                    df.at[idx, "face_area_frac"] = np.nan

            processed_count += 1
            if processed_count % save_every == 0:
                df[COLUMNS].to_csv(manifest_path, index=False)

        pbar.close()

    # Save final results
    df[COLUMNS].to_csv(manifest_path, index=False)
    logging.info(f"Face detection complete. Saved updated manifest to {manifest_path}")

    if failures:
        logging.warning(f"Total unreadable / failed images: {len(failures)}")
    else:
        logging.info("Zero image read failures.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe face detection on manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--force", action="store_true", help="Force re-running on all rows")
    parser.add_argument("--save-every", type=int, default=200, help="Save interval for checkpointing")
    parser.add_argument("--max-dim", type=int, default=1280, help="Max image edge for detection")
    args = parser.parse_args()

    run_detection(
        manifest_path=args.manifest,
        force=args.force,
        save_every=args.save_every,
        max_dim=args.max_dim,
    )


if __name__ == "__main__":
    main()
