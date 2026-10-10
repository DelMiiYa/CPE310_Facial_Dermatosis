"""SCIN metadata loading and parsing (no image I/O).

Everything SCIN-specific lives here so the rest of the pipeline only sees
clean, typed DataFrames.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCIN_DIR = PROJECT_ROOT / "data" / "raw" / "scin" / "dataset"
CASES_CSV = SCIN_DIR / "scin_cases.csv"
LABELS_CSV = SCIN_DIR / "scin_labels.csv"

# Candidate class groups -> SCIN dermatologist condition names.
# "strict" = the condition itself; "broad" adds close relatives that a
# dermatologist might list interchangeably. Final choice is made after review.
CLASS_GROUPS: dict[str, dict[str, list[str]]] = {
    "Acne": {
        "strict": ["Acne"],
        "broad": ["Acne", "Acneiform eruption"],
    },
    "Rosacea": {
        "strict": ["Rosacea"],
        "broad": ["Rosacea"],
    },
    "Eczema": {
        "strict": ["Eczema"],
        "broad": [
            "Eczema", "Infected eczema", "Lichenified eczematous dermatitis",
            "Acute dermatitis, NOS", "Acute dermatitis", "Acute and chronic dermatitis",
        ],
    },
    "Contact dermatitis": {
        "strict": ["Allergic Contact Dermatitis", "Irritant Contact Dermatitis"],
        "broad": [
            "Allergic Contact Dermatitis", "Irritant Contact Dermatitis",
            "CD - Contact dermatitis", "Contact dermatitis, NOS",
            "Photodermatitis", "Photocontact dermatitis [berloque dermatitis]",
        ],
    },
    "Seborrheic dermatitis": {
        "strict": ["Seborrheic Dermatitis"],
        "broad": ["Seborrheic Dermatitis"],
    },
    "Perioral dermatitis": {
        "strict": ["Perioral Dermatitis"],
        "broad": ["Perioral Dermatitis"],
    },
    "Folliculitis": {
        "strict": ["Folliculitis"],
        "broad": ["Folliculitis"],
    },
}


def _parse_dict(value) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def load_cases() -> pd.DataFrame:
    """One row per case: case metadata + dermatologist labels + parsed fields."""
    cases = pd.read_csv(CASES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = cases.merge(labels, on="case_id", how="left", validate="one_to_one")

    df["weighted_label"] = df["weighted_skin_condition_label"].map(_parse_dict)
    df["top_label"] = df["weighted_label"].map(lambda d: max(d, key=d.get) if d else None)
    df["top_weight"] = df["weighted_label"].map(lambda d: max(d.values()) if d else 0.0)
    df["head_or_neck"] = df["body_parts_head_or_neck"].eq("YES")

    # Dermatologist-estimated FST: most common of up to 3 labels.
    fst_cols = [f"dermatologist_fitzpatrick_skin_type_label_{i}" for i in (1, 2, 3)]
    df["efst"] = df[fst_cols].mode(axis=1, dropna=True).iloc[:, 0] if len(df) else None

    df["n_images"] = df[[f"image_{i}_path" for i in (1, 2, 3)]].notna().sum(axis=1)
    return df


def group_weight(weighted: dict[str, float], names: list[str]) -> float:
    """Total weighted-label mass assigned to any condition in `names`."""
    return sum(w for k, w in weighted.items() if k in names)


def explode_images(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image (case_id, image_path, shot_type) for the given cases."""
    rows = []
    for i in (1, 2, 3):
        part = df[["case_id", f"image_{i}_path", f"image_{i}_shot_type"]].dropna(
            subset=[f"image_{i}_path"]
        )
        part.columns = ["case_id", "image_path", "shot_type"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
