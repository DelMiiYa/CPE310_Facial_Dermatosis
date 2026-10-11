import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_roboflow_client():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment or .env file.")
    return Roboflow(api_key=api_key)

def download_rosacea(output_dir: Path = None):
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw" / "roboflow-rosacea"

    # Make it re-runnable: skip if directory already has files
    if output_dir.exists() and any(output_dir.iterdir()):
        logging.info(f"Directory {output_dir} already exists and is not empty. Skipping download.")
        return output_dir

    rf = get_roboflow_client()

    # The dataset can be accessed from the original workspace or the user's fork
    candidates = [
        ("skin-face-ixq7s", "rosacea-n6xhv-jnjno"),
        ("d3l-m11", "rosacea-n6xhv-jnjno-rruul"),
    ]

    project = None
    versions = []
    chosen_ws, chosen_proj = None, None

    for ws_name, proj_name in candidates:
        try:
            ws = rf.workspace(ws_name)
            p = ws.project(proj_name)
            vs = p.versions()
            logging.info(f"Checked {ws_name}/{proj_name}: {len(vs)} version(s) found.")
            if vs:
                project = p
                versions = vs
                chosen_ws, chosen_proj = ws_name, proj_name
                break
            elif project is None:
                project = p
                chosen_ws, chosen_proj = ws_name, proj_name
        except Exception as e:
            logging.debug(f"Could not load {ws_name}/{proj_name}: {e}")

    if not versions:
        logging.error(f"No versions found for {chosen_ws}/{chosen_proj}.")
        logging.info("Please generate a version with NO preprocessing and NO augmentation in Roboflow.")
        return None

    # Print version info
    for v in versions:
        v_num = v.version
        aug = getattr(v, "augmentation", {})
        prep = getattr(v, "preprocessing", {})
        img_count = getattr(v, "images", "unknown")
        logging.info(f"Version {v_num}: images={img_count}, preprocessing={prep}, augmentation={aug}")

    # Prefer unaugmented version
    unaugmented_versions = [v for v in versions if not getattr(v, "augmentation", {})]
    if unaugmented_versions:
        selected_version = unaugmented_versions[-1]
        logging.info(f"Selected unaugmented version: {selected_version.version}")
    else:
        selected_version = versions[-1]
        logging.warning(
            f"All versions have augmentations! Selected highest version {selected_version.version}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading version {selected_version.version} to {output_dir}...")
    dataset = selected_version.download("coco", location=str(output_dir))
    logging.info(f"Download complete: {dataset.location}")

    # Inspect images and classes
    count_summary(output_dir)
    return output_dir

def count_summary(dataset_dir: Path):
    """Count images and examine annotations in dataset_dir."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in img_exts]
    logging.info(f"Total image files found in {dataset_dir}: {len(images)}")

    json_files = list(dataset_dir.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "categories" in data:
                cats = [c.get("name") for c in data["categories"]]
                logging.info(f"Annotation file {jf.name} categories: {cats}")
        except Exception as e:
            logging.debug(f"Could not parse {jf}: {e}")

def list_face_skin_versions():
    """List versions for Task 1b (d3l-m11/face-skin-diseases-bbveb)."""
    rf = get_roboflow_client()
    ws = rf.workspace("d3l-m11")
    p = ws.project("face-skin-diseases-bbveb")
    vs = p.versions()
    logging.info(f"Project 'd3l-m11/face-skin-diseases-bbveb': {len(vs)} version(s) found.")
    for v in vs:
        logging.info(f"Version {v.version}: images={getattr(v, 'images', '?')}, preprocessing={getattr(v, 'preprocessing', {})}, augmentation={getattr(v, 'augmentation', {})}")
    return vs

if __name__ == "__main__":
    download_rosacea()
