import argparse
import html
import logging
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
GALLERIES_DIR = REPORTS_DIR / "galleries"
PATCH_SAMPLES_DIR = REPORTS_DIR / "patch_samples"

from src.manifest import load_manifest, MANIFEST_PATH
from src.detection import (
    FaceDetector,
    FaceLandmarker,
    extract_face_patches,
    extract_tile_patches,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate summary table of face detection results and write markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "step5_face_detection_summary.md"

    total = len(df)
    face_count = df["face_detected"].astype(bool).sum()
    noface_count = total - face_count

    lines = []
    lines.append("# Step 5: MediaPipe Face Detection & Analysis Summary\n")
    lines.append(f"- **Detector Model:** `blaze_face_short_range.tflite` (MediaPipe Tasks Vision)")
    lines.append(f"- **Landmarker Model:** `face_landmarker.task` (478 landmarks)")
    lines.append(f"- **Total Manifest Rows Processed:** {total:,}")
    lines.append(f"- **Total Face Detected:** {face_count:,} ({face_count / total * 100:.2f}%)")
    lines.append(f"- **Total No-Face (Close-up / Non-face):** {noface_count:,} ({noface_count / total * 100:.2f}%)\n")

    # Source x Label Table
    lines.append("## Detection Rates by Source and Label\n")
    lines.append("| Source | Label | Total | Face Detected (True) | No-Face (False) | Face Rate (%) | Median Face Area Frac |")
    lines.append("|---|---|---|---|---|---|---|")

    grouped = df.groupby(["source", "label"], observed=True)
    summary_rows = []
    for (src, lbl), grp in grouped:
        n = len(grp)
        n_face = grp["face_detected"].astype(bool).sum()
        n_noface = n - n_face
        face_rate = (n_face / n * 100) if n > 0 else 0.0
        face_rows = grp[grp["face_detected"].astype(bool)]
        med_frac = face_rows["face_area_frac"].median() if len(face_rows) > 0 else float("nan")
        med_str = f"{med_frac:.4f}" if not np.isnan(med_frac) else "N/A"
        summary_rows.append((src, lbl, n, n_face, n_noface, face_rate, med_str))
        lines.append(f"| `{src}` | `{lbl}` | {n:,} | {n_face:,} | {n_noface:,} | {face_rate:.1f}% | {med_str} |")

    lines.append("\n## Median Face Area Fraction by Source\n")
    lines.append("| Source | Total Images | Faces Detected | Median Face Area Frac | Mean Face Area Frac |")
    lines.append("|---|---|---|---|---|")
    src_grouped = df.groupby("source", observed=True)
    for src, grp in src_grouped:
        n = len(grp)
        face_rows = grp[grp["face_detected"].astype(bool)]
        n_face = len(face_rows)
        med_frac = face_rows["face_area_frac"].median() if n_face > 0 else float("nan")
        mean_frac = face_rows["face_area_frac"].mean() if n_face > 0 else float("nan")
        med_str = f"{med_frac:.4f}" if not np.isnan(med_frac) else "N/A"
        mean_str = f"{mean_frac:.4f}" if not np.isnan(mean_frac) else "N/A"
        lines.append(f"| `{src}` | {n:,} | {n_face:,} | {med_str} | {mean_str} |")

    # SCIN Fitzpatrick breakdown
    lines.append("\n## SCIN External Test Set: Breakdown by Fitzpatrick Skin Type (FST)\n")
    lines.append("| Fitzpatrick Type | Total Images | Face Detected | No-Face | Face Rate (%) | Median Face Area Frac |")
    lines.append("|---|---|---|---|---|---|")
    scin_df = df[df["source"] == "scin"]
    if len(scin_df) > 0:
        fst_grouped = scin_df.groupby("fst", observed=True, dropna=False)
        for fst_val, grp in fst_grouped:
            fst_label = str(fst_val) if pd.notna(fst_val) else "Unspecified"
            n = len(grp)
            n_face = grp["face_detected"].astype(bool).sum()
            n_noface = n - n_face
            rate = (n_face / n * 100) if n > 0 else 0.0
            face_rows = grp[grp["face_detected"].astype(bool)]
            med_frac = face_rows["face_area_frac"].median() if len(face_rows) > 0 else float("nan")
            med_str = f"{med_frac:.4f}" if not np.isnan(med_frac) else "N/A"
            lines.append(f"| {fst_label} | {n:,} | {n_face:,} | {n_noface:,} | {rate:.1f}% | {med_str} |")

    report_content = "\n".join(lines)
    report_file.write_text(report_content, encoding="utf-8")
    logging.info(f"Summary markdown written to {report_file}")
    return report_content


def generate_patch_samples_gallery(df: pd.DataFrame, num_samples: int = 12, seed: int = 42):
    """Extract sample patches for face and no-face images, save patches and HTML gallery."""
    PATCH_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    GALLERIES_DIR.mkdir(parents=True, exist_ok=True)

    face_pool = df[df["face_detected"].astype(bool)]
    noface_pool = df[~df["face_detected"].astype(bool)]

    sample_faces = face_pool.sample(n=min(num_samples, len(face_pool)), random_state=seed)
    sample_nofaces = noface_pool.sample(n=min(num_samples, len(noface_pool)), random_state=seed)

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Step 5: Patch Extraction Samples</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; color: #212529; margin: 20px; }",
        "  h1, h2 { color: #1e293b; }",
        "  .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
        "  .card-header { font-weight: bold; margin-bottom: 10px; font-size: 14px; color: #334155; }",
        "  .row { display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }",
        "  .overview-img { border: 2px solid #3b82f6; border-radius: 4px; max-width: 320px; max-height: 320px; object-fit: contain; }",
        "  .patch-container { display: flex; flex-wrap: wrap; gap: 10px; }",
        "  .patch-item { text-align: center; font-size: 11px; color: #475569; width: 140px; }",
        "  .patch-item img { width: 100%; height: 140px; object-fit: cover; border-radius: 4px; border: 1px solid #cbd5e1; }",
        "  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 5px; }",
        "  .badge-face { background: #dbeafe; color: #1e40af; }",
        "  .badge-close { background: #fef3c7; color: #92400e; }",
        "</style></head><body>",
        "<h1>Step 5: MediaPipe Patch Extraction Samples</h1>",
        "<p>This gallery validates patch extraction for both <strong>face-detected selfies</strong> (5 landmarked regions: Forehead, Nose, L Cheek, R Cheek, Chin resized to 224&times;224) and <strong>no-face close-ups</strong> (grid tiles resized to 224&times;224).</p>",
        "<h2>1. Face-Detected Selfies (Landmarked / Proportional Crops)</h2>",
    ]

    with FaceDetector() as detector, FaceLandmarker() as landmarker:
        # 1. Process Face Images
        for idx, row in sample_faces.iterrows():
            img_path = PROJECT_ROOT / row["path"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            det = detector.detect(img)
            patches_dict = extract_face_patches(img, det, landmarker)

            # Draw visual boxes on overview image
            annotated = img.copy()
            colors = {
                "forehead": (0, 165, 255),
                "nose": (0, 255, 0),
                "left_cheek": (255, 0, 0),
                "right_cheek": (255, 0, 255),
                "chin": (0, 255, 255),
            }
            for reg, (bx1, by1, bx2, by2) in patches_dict["boxes"].items():
                col = colors.get(reg, (255, 255, 255))
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), col, 3)
                cv2.putText(annotated, reg, (bx1 + 5, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # Save annotated overview
            face_sample_dir = PATCH_SAMPLES_DIR / f"face_{row['image_id']}"
            face_sample_dir.mkdir(parents=True, exist_ok=True)
            overview_path = face_sample_dir / "overview.jpg"
            # resize annotated if huge for gallery display
            ann_disp = annotated
            if max(h, w) > 640:
                s = 640.0 / max(h, w)
                ann_disp = cv2.resize(annotated, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(overview_path), ann_disp)

            # Save individual patches
            patch_html_items = []
            for reg, patch_bgr in patches_dict["patches"].items():
                p_file = face_sample_dir / f"{reg}.jpg"
                cv2.imwrite(str(p_file), patch_bgr)
                rel_patch = f"../patch_samples/face_{row['image_id']}/{reg}.jpg"
                patch_html_items.append(
                    f"<div class='patch-item'><img src='{rel_patch}' loading='lazy'><div><strong>{reg}</strong><br>224&times;224</div></div>"
                )

            rel_overview = f"../patch_samples/face_{row['image_id']}/overview.jpg"
            html_parts.append(
                f"<div class='card'>"
                f"<div class='card-header'><span class='badge badge-face'>Face Detected</span> "
                f"Source: <code>{html.escape(str(row['source']))}</code> | Label: <code>{html.escape(str(row['label']))}</code> | "
                f"ID: <code>{row['image_id']}</code> | Conf: {row['face_conf']:.3f} | Area Frac: {row['face_area_frac']:.4f}</div>"
                f"<div class='row'>"
                f"<div><img class='overview-img' src='{rel_overview}' loading='lazy'><div style='font-size:11px; text-align:center;'>Source Image with Crops</div></div>"
                f"<div class='patch-container'>{''.join(patch_html_items)}</div>"
                f"</div></div>"
            )

        html_parts.append("<h2>2. No-Face Close-ups (Grid Tiling)</h2>")

        # 2. Process No-Face Images
        for idx, row in sample_nofaces.iterrows():
            img_path = PROJECT_ROOT / row["path"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            tiles_dict = extract_tile_patches(img, target_size=224, max_tiles=9)
            tiled_img = tiles_dict["image_tiled"]
            h_t, w_t = tiled_img.shape[:2]

            annotated = tiled_img.copy()
            for tidx, (bx1, by1, bx2, by2) in enumerate(tiles_dict["boxes"]):
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                cv2.putText(annotated, f"T{tidx}", (bx1 + 5, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            noface_sample_dir = PATCH_SAMPLES_DIR / f"noface_{row['image_id']}"
            noface_sample_dir.mkdir(parents=True, exist_ok=True)
            overview_path = noface_sample_dir / "overview.jpg"
            ann_disp = annotated
            if max(h_t, w_t) > 640:
                s = 640.0 / max(h_t, w_t)
                ann_disp = cv2.resize(annotated, (int(w_t * s), int(h_t * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(overview_path), ann_disp)

            patch_html_items = []
            for tidx, patch_bgr in enumerate(tiles_dict["patches"]):
                p_file = noface_sample_dir / f"tile_{tidx}.jpg"
                cv2.imwrite(str(p_file), patch_bgr)
                rel_patch = f"../patch_samples/noface_{row['image_id']}/tile_{tidx}.jpg"
                patch_html_items.append(
                    f"<div class='patch-item'><img src='{rel_patch}' loading='lazy'><div><strong>Tile {tidx}</strong><br>224&times;224</div></div>"
                )

            rel_overview = f"../patch_samples/noface_{row['image_id']}/overview.jpg"
            html_parts.append(
                f"<div class='card'>"
                f"<div class='card-header'><span class='badge badge-close'>No Face Detected (Close-up)</span> "
                f"Source: <code>{html.escape(str(row['source']))}</code> | Label: <code>{html.escape(str(row['label']))}</code> | "
                f"ID: <code>{row['image_id']}</code> | Tiles: {len(tiles_dict['patches'])}</div>"
                f"<div class='row'>"
                f"<div><img class='overview-img' src='{rel_overview}' loading='lazy'><div style='font-size:11px; text-align:center;'>Tiled Image Grid</div></div>"
                f"<div class='patch-container'>{''.join(patch_html_items)}</div>"
                f"</div></div>"
            )

    html_parts.append("</body></html>")
    gallery_file = GALLERIES_DIR / "step5_patch_samples.html"
    gallery_file.write_text("\n".join(html_parts), encoding="utf-8")
    logging.info(f"Patch sample gallery written to {gallery_file}")


def generate_noface_galleries(df: pd.DataFrame, max_per_gallery: int = 60, seed: int = 42):
    """Generate per-source galleries of NO-FACE images for manual user review."""
    GALLERIES_DIR.mkdir(parents=True, exist_ok=True)
    noface_df = df[~df["face_detected"].astype(bool)]

    # Target sources and labels requested:
    # skin_assessment Normal/Acne/Dermatitis/Rosacea, yolo_acne, roboflow_rosacea, scin
    groups = [
        ("skin_assessment", "Normal"),
        ("skin_assessment", "Acne"),
        ("skin_assessment", "Dermatitis"),
        ("skin_assessment", "Rosacea"),
        ("yolo_acne", "Acne"),
        ("roboflow_rosacea", "Rosacea"),
        ("scin", None),  # All SCIN no-face
    ]

    for src, lbl in groups:
        if lbl is not None:
            subset = noface_df[(noface_df["source"] == src) & (noface_df["label"] == lbl)]
            file_name = f"noface_{src}_{lbl}.html"
            title = f"No-Face Review: {src} - {lbl}"
        else:
            subset = noface_df[noface_df["source"] == src]
            file_name = f"noface_{src}.html"
            title = f"No-Face Review: {src} (All Labels)"

        if len(subset) == 0:
            logging.info(f"No images found for {title}, skipping gallery.")
            continue

        sample = subset.sample(n=min(max_per_gallery, len(subset)), random_state=seed)

        html_lines = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            "<style>",
            "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f1f5f9; padding: 20px; }",
            "  h2 { color: #0f172a; margin-bottom: 5px; }",
            "  .subtitle { font-size: 13px; color: #64748b; margin-bottom: 20px; }",
            "  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }",
            "  .item { background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
            "  .item img { width: 100%; height: 200px; object-fit: contain; background: #f8fafc; border-radius: 4px; }",
            "  .caption { font-size: 11px; color: #334155; margin-top: 6px; word-break: break-all; }",
            "</style></head><body>",
            f"<h2>{title}</h2>",
            f"<div class='subtitle'>Showing {len(sample)} sample images out of {len(subset)} total no-face images. Goal: verify facial close-up vs body part.</div>",
            "<div class='gallery'>",
        ]

        for _, row in sample.iterrows():
            img_path = Path(row["path"])
            rel_img_src = f"../../../{img_path.as_posix()}"
            fst_info = f" | FST: {row['fst']}" if pd.notna(row.get("fst")) else ""
            subj_info = f"<br>Subject: {html.escape(str(row.get('subject_id', '')))}" if pd.notna(row.get("subject_id")) else ""
            html_lines.append(
                f"<div class='item'>"
                f"<img src='{rel_img_src}' loading='lazy'>"
                f"<div class='caption'><strong>{html.escape(img_path.name)}</strong>{fst_info}{subj_info}</div>"
                f"</div>"
            )

        html_lines.append("</div></body></html>")
        out_file = GALLERIES_DIR / file_name
        out_file.write_text("\n".join(html_lines), encoding="utf-8")
        logging.info(f"Generated no-face gallery: {out_file.name} ({len(sample)} images)")


def generate_tiny_face_gallery(df: pd.DataFrame, max_images: int = 60):
    """Generate gallery of face images with very small face_area_frac (< 0.05) to inspect edge cases."""
    GALLERIES_DIR.mkdir(parents=True, exist_ok=True)
    face_df = df[df["face_detected"].astype(bool) & df["face_area_frac"].notna()]
    
    # Sort by smallest face_area_frac
    tiny_faces = face_df.sort_values(by="face_area_frac", ascending=True).head(max_images)
    if len(tiny_faces) == 0:
        return

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Tiny Face Area Review</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f1f5f9; padding: 20px; }",
        "  h2 { color: #0f172a; margin-bottom: 5px; }",
        "  .subtitle { font-size: 13px; color: #64748b; margin-bottom: 20px; }",
        "  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }",
        "  .item { background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
        "  .item img { width: 100%; height: 200px; object-fit: contain; background: #f8fafc; border-radius: 4px; }",
        "  .caption { font-size: 11px; color: #334155; margin-top: 6px; word-break: break-all; }",
        "</style></head><body>",
        "<h2>Face Detected with Smallest Face Area Fraction</h2>",
        f"<div class='subtitle'>Showing {len(tiny_faces)} images with the smallest face bounding box relative to image size.</div>",
        "<div class='gallery'>",
    ]

    for _, row in tiny_faces.iterrows():
        img_path = Path(row["path"])
        rel_img_src = f"../../../{img_path.as_posix()}"
        html_lines.append(
            f"<div class='item'>"
            f"<img src='{rel_img_src}' loading='lazy'>"
            f"<div class='caption'><strong>{html.escape(img_path.name)}</strong><br>"
            f"Source: {row['source']} | Label: {row['label']}<br>"
            f"Area Frac: {row['face_area_frac']:.4f} | Conf: {row['face_conf']:.3f}</div>"
            f"</div>"
        )

    html_lines.append("</div></body></html>")
    out_file = GALLERIES_DIR / "face_tiny_area.html"
    out_file.write_text("\n".join(html_lines), encoding="utf-8")
    logging.info(f"Generated tiny face area gallery: {out_file.name} ({len(tiny_faces)} images)")


def main():
    df = load_manifest()
    logging.info("Generating summary report...")
    summary = generate_summary_report(df)
    print("\n" + summary + "\n")

    logging.info("Generating patch samples and gallery...")
    generate_patch_samples_gallery(df, num_samples=12, seed=42)

    logging.info("Generating no-face review galleries...")
    generate_noface_galleries(df, max_per_gallery=60, seed=42)

    logging.info("Generating tiny face area gallery...")
    generate_tiny_face_gallery(df, max_images=60)

    logging.info("All Step 5 reports and galleries generated successfully!")


if __name__ == "__main__":
    main()
