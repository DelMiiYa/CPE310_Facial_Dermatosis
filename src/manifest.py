import pandas as pd
from pathlib import Path
import uuid
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "manifest.csv"

COLUMNS = [
    "image_id", "source", "subject_id", "path", "orig_label", "label",
    "label_confidence", "body_part", "shot_type", "fst", "role", "phash",
    "face_detected", "review_status", "split"
]

ROLES = {"train_pool", "external_test"}
REVIEW_STATUSES = {"pending", "accepted", "rejected"}
VALID_LABELS = {"Acne", "Rosacea", "Dermatitis", "Normal"}

def init_manifest():
    """Initialize an empty manifest if it doesn't exist."""
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(MANIFEST_PATH, index=False)
        logging.info(f"Initialized empty manifest at {MANIFEST_PATH}")

def load_manifest() -> pd.DataFrame:
    """Load the manifest, preserving correct dtypes."""
    if not MANIFEST_PATH.exists():
        init_manifest()
    
    dtypes = {
        "image_id": "string",
        "source": "string",
        "subject_id": "string",
        "path": "string",
        "orig_label": "string",
        "label": "string",
        "label_confidence": "float32",
        "body_part": "string",
        "shot_type": "string",
        "fst": "string",
        "role": "string",
        "phash": "string",
        "face_detected": "boolean",
        "review_status": "string",
        "split": "string"
    }
    df = pd.read_csv(MANIFEST_PATH, dtype=dtypes)
    return df

def validate_append(new_rows: pd.DataFrame, existing_df: pd.DataFrame) -> None:
    """Validate new rows before appending to ensure schema constraints."""
    missing_cols = set(COLUMNS) - set(new_rows.columns)
    if missing_cols:
        raise ValueError(f"New rows missing columns: {missing_cols}")
    
    if new_rows["image_id"].duplicated().any():
        raise ValueError("Duplicate image_ids found in new rows.")
    
    overlap = set(new_rows["image_id"]).intersection(set(existing_df["image_id"]))
    if overlap:
        raise ValueError(f"Found {len(overlap)} image_ids already in the manifest. Example: {list(overlap)[0]}")
    
    invalid_roles = set(new_rows["role"].dropna()) - ROLES
    if invalid_roles:
        raise ValueError(f"Invalid roles: {invalid_roles}")
    
    invalid_status = set(new_rows["review_status"].dropna()) - REVIEW_STATUSES
    if invalid_status:
        raise ValueError(f"Invalid review_statuses: {invalid_status}")

def append_to_manifest(new_rows: pd.DataFrame) -> None:
    """Validate and append new rows to the manifest."""
    # Ensure all columns exist, fill missing with None
    for col in COLUMNS:
        if col not in new_rows:
            new_rows[col] = None
    
    # Reorder to match schema
    new_rows = new_rows[COLUMNS]
    
    # Fill defaults for strictly required fields where applicable
    if "review_status" in new_rows and new_rows["review_status"].isna().any():
        new_rows.loc[new_rows["review_status"].isna(), "review_status"] = "pending"

    df = load_manifest()
    validate_append(new_rows, df)
    
    df_combined = pd.concat([df, new_rows], ignore_index=True)
    df_combined.to_csv(MANIFEST_PATH, index=False)
    logging.info(f"Appended {len(new_rows)} rows to manifest.")

def generate_image_id(source: str, filename: str) -> str:
    """Generate a stable, unique ID based on source and original filename."""
    return f"{source}_{uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]}"
