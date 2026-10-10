"""Download SCIN metadata CSVs (no images) from the public GCS bucket.

Usage:
    python scripts/scin_fetch_metadata.py [--force]

Files are written to data/raw/scin/dataset/. Existing files are kept unless
--force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

BUCKET_URL = "https://storage.googleapis.com/dx-scin-public-data"
FILES = ["dataset/scin_cases.csv", "dataset/scin_labels.csv"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "scin"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(rel_path: str, force: bool) -> Path:
    dest = OUT_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to re-download)")
        return dest

    url = f"{BUCKET_URL}/{rel_path}"
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.replace(dest)  # atomic: never leave a half-written CSV behind
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    for rel in FILES:
        path = download(rel, args.force)
        print(f"       {path.stat().st_size / 1e6:.2f} MB  sha256={sha256(path)[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
