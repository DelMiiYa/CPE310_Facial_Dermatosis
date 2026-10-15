"""Step 6: Perceptual Hashing (pHash) Deduplication, Leakage Prevention & Subject Clustering.

Usage:
    python scripts/deduplicate.py

Features:
1. Computes 64-bit pHash and horizontal-flip pHash for all manifest images using ThreadPoolExecutor.
2. Applies user-approved Task 2 source filtering:
   - skin_assessment Rosacea/Dermatitis/Acne -> rejected (DermNet / non-face / surplus).
   - skin_assessment Normal: face_detected == True kept; face_detected == False rejected.
   - yolo_acne -> rejected (augmented export, surplus).
   - iga34, skin_defects, roboflow_face_skin, roboflow_rosacea -> accepted / candidates.
3. Cross-checks all candidate training images against SCIN external test set:
   - Any training image with Hamming distance <= 6 to any SCIN image is rejected (leakage prevention).
4. Collapses Roboflow augmented copies (.rf. stems):
   - Keeps 1 best representative per cluster.
   - Marks redundant copies rejected (never deletes raw files).
   - Assigns a unified subject_id to all copies in the cluster.
5. Performs pairwise pHash graph clustering (Hamming <= 6) across all accepted training images:
   - Flags and rejects any clusters with conflicting class labels.
   - Merges near-duplicates and unifies subject_ids.
6. Assigns robust subject_ids to all accepted images so Step 8 grouped splitting has zero leakage.
7. Generates data/reports/step6_deduplication_summary.md and updates data/manifest/manifest.csv.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.manifest import PROJECT_ROOT, load_manifest, MANIFEST_PATH, COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_PATH = PROJECT_ROOT / "data" / "manifest" / "phash_cache.json"
REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "step6_deduplication_summary.md"
HAMMING_THRESHOLD = 6


class UnionFind:
    """Disjoint Set Union (DSU) for clustering near-duplicates and subjects."""

    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def compute_hashes_for_image(path_str: str) -> Tuple[str, str, bool]:
    """Compute (phash, phash_flipped, success) for a given image path."""
    full_path = PROJECT_ROOT / path_str
    if not full_path.exists():
        return "", "", False
    try:
        with Image.open(full_path) as img:
            # Convert to RGB if needed (handles RGBA, P, grayscale, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            h_orig = str(imagehash.phash(img))
            h_flip = str(imagehash.phash(img.transpose(Image.FLIP_LEFT_RIGHT)))
            return h_orig, h_flip, True
    except Exception as e:
        logging.warning(f"Error hashing {path_str}: {e}")
        return "", "", False


def get_all_hashes(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load or compute pHashes for all images, caching to JSON."""
    cache = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logging.info(f"Loaded {len(cache)} cached hashes from {CACHE_PATH}")
        except Exception as e:
            logging.warning(f"Could not load cache: {e}. Recomputing.")
            cache = {}

    missing_paths = [p for p in df["path"] if p not in cache]
    if missing_paths:
        logging.info(f"Computing pHashes for {len(missing_paths)} images using ThreadPoolExecutor...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(tqdm(ex.map(compute_hashes_for_image, missing_paths), total=len(missing_paths), desc="pHash computation"))

        for p, (h_orig, h_flip, success) in zip(missing_paths, results):
            if success:
                cache[p] = {"phash": h_orig, "phash_flip": h_flip}

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        logging.info(f"Updated hash cache saved to {CACHE_PATH}")

    orig_hashes = {}
    flip_hashes = {}
    for p in df["path"]:
        if p in cache:
            orig_hashes[p] = cache[p]["phash"]
            flip_hashes[p] = cache[p]["phash_flip"]
        else:
            orig_hashes[p] = ""
            flip_hashes[p] = ""

    return orig_hashes, flip_hashes


def min_hamming_distance(h1: str, h1_flip: str, h2: str, h2_flip: str) -> int:
    """Minimum Hamming distance considering both original and horizontal flip."""
    if not h1 or not h2:
        return 999
    # imagehash.hex_to_hash
    o1 = imagehash.hex_to_hash(h1)
    f1 = imagehash.hex_to_hash(h1_flip)
    o2 = imagehash.hex_to_hash(h2)
    f2 = imagehash.hex_to_hash(h2_flip)
    return min(o1 - o2, o1 - f2, f1 - o2, f1 - f2)


def extract_normal_subject(path_str: str) -> str:
    """Parse subject identity from skin_assessment Normal filenames."""
    fname = path_str.split("/")[-1]
    # e.g., 0_0_cengyongti_0042.jpeg -> norm_cengyongti
    m = re.match(r"^0_0_([a-zA-Z0-9]+)_\d+\.jpe?g$", fname, re.IGNORECASE)
    if m:
        return f"norm_{m.group(1).lower()}"
    # e.g., aug_0_123.jpeg -> norm_aug_0
    m2 = re.match(r"^aug_(\d+)_.*\.jpe?g$", fname, re.IGNORECASE)
    if m2:
        return f"norm_aug_{m2.group(1)}"
    clean_stem = fname.split(".")[0].split(".rf.")[0]
    return f"norm_{clean_stem}"


def main():
    logging.info("Starting Step 6: Deduplication, Leakage Check & Subject Clustering...")
    df = load_manifest()
    total_initial = len(df)
    logging.info(f"Loaded manifest with {total_initial} rows.")

    # 1. Compute / fetch pHashes
    orig_hashes, flip_hashes = get_all_hashes(df)
    df["phash"] = df["path"].map(orig_hashes)

    # Track initial status counts
    before_counts = df.groupby(["source", "label"]).size().to_dict()

    # 2. Apply Task 2 dataset decisions
    # Ensure rejection_reason column exists
    if "rejection_reason" not in df.columns:
        df["rejection_reason"] = None

    for idx, row in df.iterrows():
        src = row["source"]
        lbl = row["label"]
        face = row["face_detected"]

        if src == "scin":
            # SCIN is external test set, already accepted
            df.at[idx, "role"] = "external_test"
            df.at[idx, "review_status"] = "accepted"

        elif src == "skin_assessment":
            if lbl in ("Rosacea", "Dermatitis"):
                df.at[idx, "review_status"] = "rejected"
                df.at[idx, "rejection_reason"] = "dermnet_repackaged_clinical"
            elif lbl == "Acne":
                df.at[idx, "review_status"] = "rejected"
                df.at[idx, "rejection_reason"] = "clinical_web_contamination_surplus"
            elif lbl == "Normal":
                if face is True:
                    df.at[idx, "review_status"] = "accepted"
                else:
                    df.at[idx, "review_status"] = "rejected"
                    df.at[idx, "rejection_reason"] = "non_face_or_no_face_detected"

        elif src == "yolo_acne":
            df.at[idx, "review_status"] = "rejected"
            df.at[idx, "rejection_reason"] = "augmented_surplus_acne"

        elif src in ("iga34", "skin_defects", "roboflow_face_skin"):
            df.at[idx, "review_status"] = "accepted"

        elif src == "roboflow_rosacea":
            # Candidate for training; will be deduplicated below
            df.at[idx, "review_status"] = "accepted"

    # 3. SCIN External Test Leakage Prevention
    logging.info("Cross-checking training candidates against SCIN external test set...")
    scin_df = df[df["role"] == "external_test"]
    scin_hashes = {
        row["image_id"]: (orig_hashes[row["path"]], flip_hashes[row["path"]])
        for _, row in scin_df.iterrows()
    }

    train_candidates_mask = (df["role"] == "train_pool") & (df["review_status"] == "accepted")
    scin_leaks_found = []

    for idx in df[train_candidates_mask].index:
        p = df.at[idx, "path"]
        img_id = df.at[idx, "image_id"]
        h1, f1 = orig_hashes[p], flip_hashes[p]
        for scin_id, (h2, f2) in scin_hashes.items():
            dist = min_hamming_distance(h1, f1, h2, f2)
            if dist <= HAMMING_THRESHOLD:
                df.at[idx, "review_status"] = "rejected"
                df.at[idx, "rejection_reason"] = f"scin_leakage_match_{scin_id}_dist_{dist}"
                scin_leaks_found.append({
                    "train_image_id": img_id,
                    "train_path": p,
                    "scin_image_id": scin_id,
                    "distance": dist
                })
                break

    logging.info(f"SCIN leakage checks complete: {len(scin_leaks_found)} matches caught and rejected.")

    # 4. Collapse Roboflow Augmented Copies in roboflow_rosacea
    logging.info("Collapsing Roboflow augmented copies (.rf. stems) in roboflow_rosacea...")
    rf_rosacea_mask = (df["source"] == "roboflow_rosacea") & (df["review_status"] == "accepted")
    rf_df = df[rf_rosacea_mask].copy()

    # Extract base stem before '.rf.'
    def get_rf_stem(p_str: str) -> str:
        fname = p_str.split("/")[-1]
        if ".rf." in fname:
            return fname.split(".rf.")[0]
        return fname.split(".")[0]

    rf_df["base_stem"] = rf_df["path"].apply(get_rf_stem)
    rf_groups = rf_df.groupby("base_stem")

    rf_dedup_before = len(rf_df)
    rf_dedup_after = 0
    rf_rejected_copies = 0

    for stem, group in rf_groups:
        # Pick best representative: prefer face_detected == True, highest face_area_frac, highest face_conf
        sorted_indices = group.sort_values(
            by=["face_detected", "face_area_frac", "face_conf"],
            ascending=[False, False, False]
        ).index.tolist()

        rep_idx = sorted_indices[0]
        rep_id = df.at[rep_idx, "image_id"]
        cluster_subj = f"rf_rosacea_{stem}"

        # Assign subject_id to all members of this cluster
        for idx in sorted_indices:
            df.at[idx, "subject_id"] = cluster_subj

        # Rep remains accepted
        df.at[rep_idx, "review_status"] = "accepted"
        rf_dedup_after += 1

        # Reject duplicate copies
        for dup_idx in sorted_indices[1:]:
            df.at[dup_idx, "review_status"] = "rejected"
            df.at[dup_idx, "rejection_reason"] = f"augmented_duplicate_of_{rep_id}"
            rf_rejected_copies += 1

    logging.info(f"roboflow_rosacea: reduced from {rf_dedup_before} images to {rf_dedup_after} distinct originals ({rf_rejected_copies} duplicates rejected).")

    # 5. General pHash Deduplication Across ALL Active Training Candidates
    logging.info("Running global pHash pairwise deduplication across active training candidates...")
    active_mask = (df["role"] == "train_pool") & (df["review_status"] == "accepted")
    active_df = df[active_mask].copy()
    active_indices = active_df.index.tolist()

    uf = UnionFind()
    pairs_under_threshold = []
    label_conflicts = []

    # Fast pairwise pHash comparison
    active_hashes = [
        (idx, orig_hashes[df.at[idx, "path"]], flip_hashes[df.at[idx, "path"]], df.at[idx, "label"], df.at[idx, "source"], df.at[idx, "subject_id"])
        for idx in active_indices
    ]

    for i in range(len(active_hashes)):
        idx1, h1, f1, lbl1, src1, subj1 = active_hashes[i]
        for j in range(i + 1, len(active_hashes)):
            idx2, h2, f2, lbl2, src2, subj2 = active_hashes[j]
            dist = min_hamming_distance(h1, f1, h2, f2)
            if dist <= HAMMING_THRESHOLD:
                pairs_under_threshold.append((idx1, idx2, dist))
                uf.union(idx1, idx2)

    # Group into connected component clusters
    clusters: Dict[int, List[int]] = {}
    for idx in active_indices:
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    multi_clusters = [members for members in clusters.values() if len(members) > 1]
    logging.info(f"Found {len(multi_clusters)} near-duplicate clusters (Hamming <= {HAMMING_THRESHOLD}).")

    global_dedup_rejected = 0

    for cluster in multi_clusters:
        # Check for label conflict
        labels_in_cluster = {df.at[idx, "label"] for idx in cluster}
        if len(labels_in_cluster) > 1:
            # Conflicting label near-duplicates: reject all to prevent noisy learning
            for idx in cluster:
                df.at[idx, "review_status"] = "rejected"
                df.at[idx, "rejection_reason"] = f"label_conflict_near_duplicate_{labels_in_cluster}"
            label_conflicts.append({
                "cluster_size": len(cluster),
                "labels": list(labels_in_cluster),
                "image_ids": [df.at[idx, "image_id"] for idx in cluster],
                "paths": [df.at[idx, "path"] for idx in cluster]
            })
            continue

        # Same label cluster:
        # If they already share an explicit multi-view subject (e.g. iga34 front/left/right of same person), keep all views!
        # Otherwise, keep 1 best representative and collapse duplicates.
        existing_subjs = {df.at[idx, "subject_id"] for idx in cluster if pd.notna(df.at[idx, "subject_id"])}
        sources = {df.at[idx, "source"] for idx in cluster}

        # If it's pure iga34 or pure skin_defects with same subject_id, they are different angles
        if len(existing_subjs) == 1 and sources.issubset({"iga34", "skin_defects"}):
            # Keep both/all views, they belong to the same confirmed subject
            continue

        # Pick best representative
        sorted_members = sorted(
            cluster,
            key=lambda idx: (
                df.at[idx, "face_detected"] is True,
                float(df.at[idx, "face_area_frac"]) if pd.notna(df.at[idx, "face_area_frac"]) else 0.0,
                float(df.at[idx, "face_conf"]) if pd.notna(df.at[idx, "face_conf"]) else 0.0
            ),
            reverse=True
        )

        rep_idx = sorted_members[0]
        rep_id = df.at[rep_idx, "image_id"]

        # Derive a shared subject ID for this cluster
        if existing_subjs:
            shared_subj = sorted(list(existing_subjs))[0]
        else:
            shared_subj = f"cluster_{df.at[rep_idx, 'label']}_{rep_id}"

        df.at[rep_idx, "subject_id"] = shared_subj
        df.at[rep_idx, "review_status"] = "accepted"

        for dup_idx in sorted_members[1:]:
            df.at[dup_idx, "subject_id"] = shared_subj
            df.at[dup_idx, "review_status"] = "rejected"
            df.at[dup_idx, "rejection_reason"] = f"near_duplicate_of_{rep_id}"
            global_dedup_rejected += 1

    # 6. Assign subject_id to all remaining accepted images
    logging.info("Assigning clean subject_ids to remaining accepted images...")
    accepted_mask = (df["role"] == "train_pool") & (df["review_status"] == "accepted")

    for idx in df[accepted_mask].index:
        current_subj = df.at[idx, "subject_id"]
        if pd.isna(current_subj) or str(current_subj).strip() == "":
            src = df.at[idx, "source"]
            p = df.at[idx, "path"]
            img_id = df.at[idx, "image_id"]
            if src == "skin_assessment" and df.at[idx, "label"] == "Normal":
                df.at[idx, "subject_id"] = extract_normal_subject(p)
            elif src == "roboflow_face_skin":
                fname_stem = p.split("/")[-1].split(".rf.")[0].split(".")[0]
                df.at[idx, "subject_id"] = f"rfs_{fname_stem}"
            elif src == "roboflow_rosacea":
                fname_stem = p.split("/")[-1].split(".rf.")[0].split(".")[0]
                df.at[idx, "subject_id"] = f"rf_ros_{fname_stem}"
            else:
                df.at[idx, "subject_id"] = f"{src}_{img_id}"

    # Also assign subject_id to rejected images missing it (for consistency)
    for idx in df[df["subject_id"].isna()].index:
        src = df.at[idx, "source"]
        img_id = df.at[idx, "image_id"]
        df.at[idx, "subject_id"] = f"{src}_{img_id}"

    # 7. Save updated manifest
    # Reorder columns matching schema
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    df.to_csv(MANIFEST_PATH, index=False)
    logging.info(f"Manifest successfully updated and saved to {MANIFEST_PATH}")

    # 8. Compile Summary Statistics & Generate Report
    accepted_train = df[(df["role"] == "train_pool") & (df["review_status"] == "accepted")]
    external_test = df[df["role"] == "external_test"]

    train_class_counts = accepted_train["label"].value_counts().to_dict()
    train_source_counts = accepted_train.groupby(["source", "label"]).size().to_dict()
    train_subjects_per_class = accepted_train.groupby("label")["subject_id"].nunique().to_dict()
    test_class_counts = external_test["label"].value_counts().to_dict()

    report_lines = [
        "# Step 6: Perceptual Deduplication & Source Filtering Summary",
        "",
        f"- **Total Rows in Manifest:** {len(df)}",
        f"- **Hamming Distance Threshold:** $\le {HAMMING_THRESHOLD}$",
        f"- **SCIN External Test Images:** {len(external_test)}",
        f"- **Accepted Training Images:** {len(accepted_train)}",
        f"- **Rejected Images (Preserved in Manifest, Not Deleted):** {len(df) - len(accepted_train) - len(external_test)}",
        "",
        "## Final Curated Training Pool (Role: train_pool, review_status: accepted)",
        "",
        "| Class | Images | Unique Subjects (`subject_id`) | Primary Sources |",
        "|---|---|---|---|",
    ]

    for lbl in ["Acne", "Rosacea", "Normal", "Dermatitis"]:
        n_imgs = train_class_counts.get(lbl, 0)
        n_subjs = train_subjects_per_class.get(lbl, 0)
        srcs = [f"{s} ({c})" for (s, l), c in train_source_counts.items() if l == lbl]
        src_str = ", ".join(srcs) if srcs else "None"
        report_lines.append(f"| **{lbl}** | **{n_imgs}** | **{n_subjs}** | {src_str} |")

    report_lines.extend([
        "",
        "## SCIN External Test Set (Role: external_test, review_status: accepted)",
        "",
        "| Class | Images | Unique Subjects | Notes |",
        "|---|---|---|---|",
        f"| Acne | {test_class_counts.get('Acne', 0)} | {external_test[external_test['label'] == 'Acne']['subject_id'].nunique()} | Held out strictly for test |",
        f"| Rosacea | {test_class_counts.get('Rosacea', 0)} | {external_test[external_test['label'] == 'Rosacea']['subject_id'].nunique()} | Held out strictly for test |",
        f"| Dermatitis | {test_class_counts.get('Dermatitis', 0)} | {external_test[external_test['label'] == 'Dermatitis']['subject_id'].nunique()} | Held out strictly for test |",
        "",
        "## Deduplication & Quality Filter Details",
        "",
        f"1. **Roboflow Augmented Copies in `roboflow_rosacea`:**",
        f"   - Scanned: {rf_dedup_before} images",
        f"   - Unique original stems kept: **{rf_dedup_after}**",
        f"   - Redundant augmented copies rejected: **{rf_rejected_copies}**",
        "",
        f"2. **SCIN External Test Leakage Prevention:**",
        f"   - Matches with Hamming $\le {HAMMING_THRESHOLD}$ against SCIN: **{len(scin_leaks_found)}**",
    ])

    if scin_leaks_found:
        report_lines.append("   - Leaks rejected:")
        for leak in scin_leaks_found:
            report_lines.append(f"     - Train image `{leak['train_image_id']}` matches SCIN `{leak['scin_image_id']}` (dist: {leak['distance']})")
    else:
        report_lines.append("   - **Zero leaks:** None of the training candidates matched any of the 337 SCIN external test images.")

    report_lines.extend([
        "",
        f"3. **Cross-Source Near-Duplicate & Conflict Analysis:**",
        f"   - Near-duplicate clusters found (Hamming $\le {HAMMING_THRESHOLD}$): {len(multi_clusters)}",
        f"   - Duplicate images rejected across active training pool: {global_dedup_rejected}",
        f"   - Label conflicts detected: **{len(label_conflicts)}**",
    ])

    if label_conflicts:
        report_lines.append("   - Conflicting clusters rejected:")
        for conf in label_conflicts:
            report_lines.append(f"     - Conflicting labels {conf['labels']} across {conf['cluster_size']} images: {conf['paths']}")
    else:
        report_lines.append("   - **Zero label conflicts detected across near-duplicate clusters.**")

    report_lines.extend([
        "",
        "## Review Status Breakdown in Manifest",
        "",
        "| Source | Label | Accepted | Rejected | Total |",
        "|---|---|---|---|---|",
    ])

    summary_group = df.groupby(["source", "label", "review_status"]).size().unstack(fill_value=0)
    for (src, lbl), r_row in summary_group.iterrows():
        acc = r_row.get("accepted", 0)
        rej = r_row.get("rejected", 0)
        report_lines.append(f"| `{src}` | `{lbl}` | {acc} | {rej} | {acc + rej} |")

    report_lines.append("")
    report_text = "\n".join(report_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Summary report written to {REPORT_PATH}")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
