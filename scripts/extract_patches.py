"""Pre-extract 224x224 facial and tile patches for all active dataset splits.

Extracts:
- Selfies (face_detected=True): 5 landmarked / proportional regions (forehead, nose, left cheek, right cheek, chin)
- Close-ups (face_detected=False): 1-9 grid tiles

Saves patches to data/processed/patches/<split>/ and records metadata to data/manifest/patches_manifest.csv.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import load_manifest, MANIFEST_PATH
from src.detection import (
    FaceDetector,
    FaceLandmarker,
    extract_face_patches,
    extract_tile_patches,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "patches"
PATCHES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "patches_manifest.csv"

ACTIVE_SPLITS = ("train", "val", "test", "external_test")


def extract_all_patches(
    output_dir: Path = PROCESSED_DIR,
    output_manifest: Path = PATCHES_MANIFEST_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Extract patches for all active rows in manifest.csv and save to disk."""
    df = load_manifest()
    active_mask = df["split"].isin(ACTIVE_SPLITS)
    active_df = df[active_mask].copy()
    logging.info(f"Loaded {len(active_df)} active images from {MANIFEST_PATH} across splits: {ACTIVE_SPLITS}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for s in ACTIVE_SPLITS:
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    # Check for existing patch manifest for resumability
    existing_patches: dict[str, dict] = {}
    if output_manifest.exists() and not force:
        try:
            existing_df = pd.read_csv(output_manifest)
            logging.info(f"Found existing patches manifest with {len(existing_df)} patches.")
            for _, r in existing_df.iterrows():
                p_path = PROJECT_ROOT / r["patch_path"]
                if p_path.exists():
                    existing_patches[r["patch_id"]] = r.to_dict()
            logging.info(f"Verified {len(existing_patches)} existing patch files on disk.")
        except Exception as e:
            logging.warning(f"Could not load existing patch manifest: {e}. Starting fresh.")

    # Group already processed images
    processed_image_ids = set()
    if existing_patches:
        for p_info in existing_patches.values():
            processed_image_ids.add(p_info["image_id"])
        logging.info(f"{len(processed_image_ids)} images already have extracted patches.")

    to_process_df = active_df if force else active_df[~active_df["image_id"].isin(processed_image_ids)]
    logging.info(f"Images to process: {len(to_process_df)}")

    new_patch_records: list[dict] = []

    with FaceDetector() as detector, FaceLandmarker() as landmarker:
        pbar = tqdm(to_process_df.iterrows(), total=len(to_process_df), desc="Extracting patches", unit="img")
        for _, row in pbar:
            img_id = row["image_id"]
            split = row["split"]
            label = row["label"]
            subj_id = row["subject_id"]
            src = row["source"]
            is_face = bool(row["face_detected"])

            img_path = PROJECT_ROOT / row["path"]
            img = cv2.imread(str(img_path))
            if img is None:
                logging.error(f"Failed to read image {img_path}")
                continue

            patches_to_save: list[tuple[str, any]] = []  # (patch_suffix, image_bgr)

            if is_face:
                try:
                    det = detector.detect(img)
                    if det.get("face_detected", False):
                        res = extract_face_patches(img, det, landmarker, target_size=224)
                        for reg_name, p_img in res["patches"].items():
                            patches_to_save.append((reg_name, p_img))
                    else:
                        # Fallback to tiles if face not detected on this pass
                        tile_res = extract_tile_patches(img, target_size=224, max_tiles=9)
                        for t_idx, p_img in enumerate(tile_res["patches"]):
                            patches_to_save.append((f"tile_{t_idx}", p_img))
                except Exception as e:
                    # Safe fallback to tiles
                    tile_res = extract_tile_patches(img, target_size=224, max_tiles=9)
                    for t_idx, p_img in enumerate(tile_res["patches"]):
                        patches_to_save.append((f"tile_{t_idx}", p_img))
            else:
                tile_res = extract_tile_patches(img, target_size=224, max_tiles=9)
                for t_idx, p_img in enumerate(tile_res["patches"]):
                    patches_to_save.append((f"tile_{t_idx}", p_img))

            # Save patches
            for patch_suffix, p_bgr in patches_to_save:
                patch_id = f"{img_id}_{patch_suffix}"
                rel_patch_path = Path("data") / "processed" / "patches" / split / f"{patch_id}.jpg"
                abs_patch_path = PROJECT_ROOT / rel_patch_path

                cv2.imwrite(str(abs_patch_path), p_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                new_patch_records.append({
                    "patch_id": patch_id,
                    "image_id": img_id,
                    "subject_id": subj_id,
                    "source": src,
                    "split": split,
                    "label": label,
                    "patch_type": patch_suffix,
                    "is_face_crop": is_face,
                    "patch_path": str(rel_patch_path).replace("\\", "/"),
                })

    all_records = list(existing_patches.values()) + new_patch_records
    patches_df = pd.DataFrame(all_records)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    patches_df.to_csv(output_manifest, index=False)
    logging.info(f"Patches manifest saved to {output_manifest} with {len(patches_df)} total patches.")

    # Summary report
    print("\n" + "=" * 60)
    print("PATCH EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Patches: {len(patches_df)}")
    print("\nPatches by Split:")
    print(patches_df["split"].value_counts().to_string())
    print("\nPatches by Label (in train):")
    print(patches_df[patches_df["split"] == "train"]["label"].value_counts().to_string())
    print("\nPatches by Type:")
    print(patches_df["patch_type"].value_counts().head(10).to_string())
    print("=" * 60 + "\n")

    return patches_df


def main():
    parser = argparse.ArgumentParser(description="Extract patches for active dataset splits")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of all patches")
    args = parser.parse_args()

    extract_all_patches(force=args.force)


if __name__ == "__main__":
    main()
