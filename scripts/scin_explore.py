"""Read-only exploration of SCIN metadata for facial (head/neck) cases.

Usage:
    python scripts/scin_explore.py

Prints a report and writes it to data/reports/scin_explore.md. Writes no
training data and downloads nothing.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sources.scin import (  # noqa: E402
    CLASS_GROUPS, PROJECT_ROOT, explode_images, group_weight, load_cases,
)

THRESHOLDS = [0.5, 0.6, 0.7]
REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "scin_explore.md"


def section(title: str) -> None:
    print(f"\n## {title}\n")


def table(df: pd.DataFrame) -> None:
    try:
        print(df.to_markdown())
    except ImportError:  # optional 'tabulate' dependency missing
        print("```\n" + df.to_string() + "\n```")
    print()


def report() -> None:
    df = load_cases()
    hn = df[df["head_or_neck"]].copy()

    print("# SCIN exploration report\n")
    section("Overview")
    print(f"- Total cases: {len(df)}  (images: {int(df['n_images'].sum())})")
    print(f"- HEAD_OR_NECK cases: {len(hn)}  (images: {int(hn['n_images'].sum())})")
    print(f"- HEAD_OR_NECK cases with no dermatologist label: {(hn['top_label'].isna()).sum()}")
    only_hn = hn[[c for c in df.columns if c.startswith("body_parts_")]].eq("YES").sum(axis=1).eq(1)
    print(f"- HEAD_OR_NECK as the *only* body part: {int(only_hn.sum())}")

    section("Top-20 top-1 labels among labelled HEAD_OR_NECK cases")
    table(hn["top_label"].value_counts().head(20).rename("cases").to_frame())

    section("Candidate classes: cases (images) by label-weight threshold, HEAD_OR_NECK only")
    print("Weight = sum of the weighted-label mass assigned to the group's condition names.\n")
    rows = []
    for cls, variants in CLASS_GROUPS.items():
        for variant, names in variants.items():
            w = hn["weighted_label"].map(lambda d: group_weight(d, names))
            row = {"class": cls, "variant": variant}
            for t in THRESHOLDS:
                sel = hn[w >= t]
                row[f">= {t}"] = f"{len(sel)} ({int(sel['n_images'].sum())})"
            row["any mention"] = f"{int((w > 0).sum())}"
            rows.append(row)
    table(pd.DataFrame(rows).set_index(["class", "variant"]))

    section("Same classes, ALL body parts (for reference: most are not facial)")
    rows = []
    for cls, variants in CLASS_GROUPS.items():
        names = variants["broad"]
        w = df["weighted_label"].map(lambda d: group_weight(d, names))
        rows.append({"class": cls, ">= 0.5 cases": int((w >= 0.5).sum())})
    table(pd.DataFrame(rows).set_index("class"))

    section("Shot types for HEAD_OR_NECK images (strict classes, weight >= 0.5)")
    rows = []
    for cls, variants in CLASS_GROUPS.items():
        w = hn["weighted_label"].map(lambda d: group_weight(d, variants["strict"]))
        imgs = explode_images(hn[w >= 0.5])
        counts = imgs["shot_type"].value_counts()
        rows.append({"class": cls, **counts.to_dict()})
    table(pd.DataFrame(rows).set_index("class").fillna(0).astype(int))

    section("Dermatologist-estimated Fitzpatrick type, HEAD_OR_NECK (strict, weight >= 0.5)")
    rows = []
    for cls, variants in CLASS_GROUPS.items():
        w = hn["weighted_label"].map(lambda d: group_weight(d, variants["strict"]))
        rows.append({"class": cls, **hn[w >= 0.5]["efst"].value_counts().to_dict()})
    table(pd.DataFrame(rows).set_index("class").fillna(0).astype(int).sort_index(axis=1))

    section("Healthy / normal skin")
    print(f"- related_category == LOOKS_HEALTHY (all body parts): {(df['related_category'] == 'LOOKS_HEALTHY').sum()}")
    print(f"- ... and HEAD_OR_NECK: {(hn['related_category'] == 'LOOKS_HEALTHY').sum()}")
    print("- Note: SCIN has no dermatologist 'healthy' label; ungradable cases are not reliable negatives.")

    section("Images per case (HEAD_OR_NECK)")
    table(hn["n_images"].value_counts().sort_index().rename("cases").to_frame())


def main() -> int:
    buf = io.StringIO()
    with redirect_stdout(buf):
        report()
    text = buf.getvalue()
    print(text)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\n[saved] {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
