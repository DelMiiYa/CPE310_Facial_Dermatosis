import sys
import pandas as pd
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.manifest import load_manifest

GALLERY_DIR = PROJECT_ROOT / "data" / "reports" / "galleries"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def generate_html_gallery(df: pd.DataFrame, source: str, label: str, output_path: Path):
    html = [
        f"<!DOCTYPE html>",
        f"<html>",
        f"<head>",
        f"<title>Gallery: {source} - {label}</title>",
        f"<style>",
        f"  body {{ font-family: sans-serif; background: #f0f0f0; }}",
        f"  .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}",
        f"  .item {{ background: white; padding: 5px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px; }}",
        f"  .item img {{ width: 100%; height: 200px; object-fit: contain; }}",
        f"  .caption {{ font-size: 12px; word-wrap: break-word; text-align: center; margin-top: 5px; }}",
        f"</style>",
        f"</head>",
        f"<body>",
        f"<h2>{source} - {label} (Sample of {len(df)})</h2>",
        f"<div class='gallery'>"
    ]
    
    for _, row in df.iterrows():
        # path is relative to project root
        img_path = Path(row['path'])
        # Since HTML is in data/reports/galleries, to get to project root is ../../../
        rel_img_src = f"../../../{img_path.as_posix()}"
        
        html.append(
            f"<div class='item'>"
            f"<img src='{rel_img_src}' loading='lazy'>"
            f"<div class='caption'>{img_path.name}<br>{row.get('subject_id', '')}</div>"
            f"</div>"
        )
        
    html.append("</div></body></html>")
    
    output_path.write_text("\n".join(html), encoding="utf-8")

def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    
    groups = df.groupby(['source', 'label'])
    for (source, label), group in groups:
        sample = group.sample(n=min(60, len(group)), random_state=42)
        safe_source = str(source).replace(' ', '_').replace('/', '_')
        safe_label = str(label).replace(' ', '_').replace('/', '_')
        
        output_file = GALLERY_DIR / f"{safe_source}_{safe_label}.html"
        generate_html_gallery(sample, str(source), str(label), output_file)
        logging.info(f"Generated gallery for {source} - {label} ({len(sample)} images) at {output_file.name}")

if __name__ == "__main__":
    main()
