# Handoff Prompt: Facial Dermatosis Classification (CPE310)

*Paste this whole file into the next AI assistant. It's self-contained.*

---

## Your role

You are an expert ML engineer continuing an **existing** end-to-end PyTorch computer-vision project. A previous assistant did the analysis and setup below. **Do not redo it, and do not reverse decisions marked as approved** unless the user asks.

**Working rules (strict):**
1. Work step by step. **Propose each step and wait for the user's approval before running it.** Never write the whole codebase at once.
2. Never delete or modify anything in `data/raw/`. Never use DermNet or clinical-only imagery for training.
3. Commit to the local git repo after each completed step (`data/` is git-ignored).
4. Stop at every **⏸ STOP** point below and let the user review.
5. Read `agent_guidelines.md` and `project_prompt.md` in the project root. They contain hard rules (domain gap, manual review, pHash dedup before splitting, 70/15/15 stratified split, Macro-F1, ResNet-50 baseline first, 3 seeds, Grad-CAM).

---

## Project context

- **Goal:** a multi-class classifier for facial skin conditions from **smartphone photos**. It must accept both **full-face selfies and close-ups**.
- **Project root:** `E:\CPE310FacialDermatorsisProject` (Windows, PowerShell, Python 3.13).
- **Hardware:** NVIDIA RTX 3050, **4 GB VRAM**. Use mixed precision (`torch.amp.autocast`) and batch size 16–32. DataLoaders need `pin_memory=True`, `persistent_workers=True` and `num_workers=4`. Install PyTorch with the explicit CUDA index URL and verify `torch.cuda.is_available()`.
- **Literature survey:** `Literature_and_Papers/Literature Survey.pdf` (written in Thai). Key takeaways:
  - The main gap is that most prior work is binary; the hard real-world task is telling **acne vs. rosacea** apart (Papers 1, 2, 5).
  - Add atopic/contact dermatitis (Papers 3, 4).
  - Backbone: **EfficientNet-B0 / ResNet-50** with transfer learning.
  - Evaluate across **Fitzpatrick skin types**.
  - Compare a single model vs. an **ensemble** (Paper 2, FACES).
  - **Patch-based** input beats whole-face input (Paper 5: forehead/nose/cheek crops). Paper 3 segments the lesion before classifying.
- **Lesson from v1 (critical):** a model trained on DermNet clinical images got 95% accuracy but failed completely on real selfies (domain shift). Training data must look like smartphone photos.

---

## ✅ Approved decisions

1. **Classes:** `Acne`, `Rosacea`, `Dermatitis` (atopic eczema + contact dermatitis merged), `Normal`. Seborrheic and perioral dermatitis are excluded (too few examples). The merge should be justified in the final report.
2. **SCIN is the external test set only. It is never used for training, validation or tuning.**
3. **Architecture:** one patch classifier plus a pretrained face detector (MediaPipe) that decides how to crop the input:
   - Face detected (selfie): crop skin regions (forehead, nose, left/right cheek, chin) and classify each patch.
   - No face (close-up): tile the image into patches and classify each.
   - Aggregate the patch scores into one diagnosis, and produce a Grad-CAM heatmap.
   - Rationale: data is scarce, so two separate models would split it; patches from both input types look alike.
4. **Pipeline design:** a single `data/manifest/manifest.csv` is the source of truth. Images stay where they are and manifest rows point to them. Every later step (face filter, dedup, review, split) reads and updates the manifest instead of copying folders.
5. Old files `download_data.py`, `filter_and_dedup.py` and `data/processed/` are **legacy and must not be used**. The old filter script pulled in DermNet folders, exact-match pHash missed augmented copies, it skipped the selfie datasets because of their folder layout, and it threw away which source each image came from.

---

## Current state of the repo (git: 2 commits on `main`)

```
.gitignore                      # ignores data/, venvs, *.pt, runs/
requirements.txt                # pandas, requests, tqdm, Pillow, imagehash, opencv-python, pymupdf
agent_guidelines.md, project_prompt.md, datasets_links.txt
Literature_and_Papers/          # survey PDF + 5 papers + pptx
src/sources/scin.py             # SCIN loader: load_cases(), group_weight(), explode_images(), CLASS_GROUPS
scripts/scin_fetch_metadata.py  # downloads scin_cases.csv + scin_labels.csv (DONE)
scripts/scin_explore.py         # read-only report → data/reports/scin_explore.md (DONE)
download_data.py, filter_and_dedup.py   # LEGACY, do not use
```

### SCIN facts (already verified)
- Public GCS bucket: images download over plain HTTPS at `https://storage.googleapis.com/dx-scin-public-data/<image_path>`, where `image_path` looks like `dataset/images/<id>.png`. No login needed.
- The CSVs are at `data/raw/scin/dataset/`. Column names are **lowercase** (e.g. `body_parts_head_or_neck == "YES"`). The label-name column is `dermatologist_skin_condition_on_label_name`, and `weighted_skin_condition_label` is a dict stored as a string.
- 5,033 cases / 10,407 images. **737 are HEAD_OR_NECK** (1,581 images), and 321 of those have no dermatologist label.
- Facial cases with label weight ≥ 0.5: Acne 26 cases (60 images), Rosacea 16 (34), Eczema-broad 37 (75), Contact-dermatitis-broad 56 (131). Rosacea is almost all FST 1–2.
- Each case has up to 3 images (one close-up, plus optional angled and at-distance shots). **All images from one case must stay together.**
- Known issues: 15 duplicate images, 1 missing file.

### Other local sources in `data/raw/` (all untouched)
| Source | Content | Size | Notes |
|---|---|---|---|
| `skin-problems-34-on-the-iga-scale/images/<person>/{1,2,3}.jpg` | phone selfies, acne IGA 3–4 | 229 people / 687 images | good domain match; acne only; `df.csv` lists paths only |
| `acne-dataset-in-yolov8-format/data-2/{train,valid,test}/{images,labels}` | acne + YOLO boxes | 927 images | look at samples first |
| `skin-assessment-dataset/skin_assesment_dataset/{acne,Rosacea,Eczema,normal,...}` | 16 classes, ~1.1k each | ~4.8k relevant | **origin unknown**, may be scraped or DermNet; look at samples first |
| `skin-defects-acne-redness-and-bags-under-the-eyes/files/{acne,redness,bags}/<person>/{front,left_side,right_side}.jpg` | selfies | 10 people per class | tiny |
| `facial-skin-diseases-dataset/` | TFRecord detection set | ? | not inspected |
| `massive-skin-disease-balanced-dataset/` | DermNet, augmented to ~7k per class | — | **EXCLUDED** |
| Roboflow `face-skin-diseases` | facial | ? | needs a free API key (ask the user) |

**The main bottleneck is phone-style photos for Rosacea and Normal.** Don't pick a fix (Roboflow, the skin-assessment set, or new collection) until the user has reviewed sample images.

---

## Next task: Step 4 (propose it to the user, then build after approval)

### 4A. Manifest + SCIN external test set
- Create `src/manifest.py` with a schema, read/write/validate helpers, and an append that never creates duplicate `image_id`s. Columns:
  `image_id, source, subject_id, path, orig_label, label, label_confidence, body_part, shot_type, fst, role, phash, face_detected, review_status, split`
  - `role` ∈ {`train_pool`, `external_test`}
  - `review_status` ∈ {`pending`, `accepted`, `rejected`}
- Add SCIN facial cases (HEAD_OR_NECK) for Acne / Rosacea / Dermatitis (Eczema-broad + Contact-dermatitis-broad from `CLASS_GROUPS`) with label weight ≥ 0.5. Mark them `role=external_test` and set `subject_id=case_id`. That's about 120 cases / about 270 images. If a case qualifies for two classes, flag it and don't assign it silently.
- Download **only those images** to `data/raw/scin/dataset/images/`. Make the download resumable (skip files that already exist, write to `.part` then rename) and log failures.

### 4B. Inventory the local sources (nothing copied or deleted)
- One loader per source in `src/sources/` that adds manifest rows pointing at the existing files, with `role=train_pool`. Cover IGA34 (Acne, `subject_id` = person folder), YOLO-acne, skin-assessment (acne/Rosacea/Eczema/normal only) and skin-defects (acne only; `redness` is not a diagnosis).

### 4C. Review galleries (⏸ STOP)
- Generate one HTML contact sheet per source × class (about 60 random thumbnails each, fixed seed) in `data/reports/galleries/`.
- Ask the user to judge for each source: does it look like real phone photos, and are the labels right? Then decide together which sources to trust, how to fill the Rosacea/Normal gap, and whether to get a Roboflow key.

### Later steps (outline only; each needs approval)
5. Face detection (MediaPipe) and region/patch cropping; record `face_detected` in the manifest.
6. Near-duplicate dedup using pHash **Hamming distance ≤ ~6–8** (not exact match), across all sources and **before** splitting. Flag duplicates that appear under different classes instead of silently keeping one.
7. Manual review of a representative sample (⏸ STOP).
8. Split 70/15/15, stratified by class and **grouped by `subject_id`**. SCIN stays external.
9. ResNet-50 baseline with weighted cross-entropy, judged on Macro-F1. Then EfficientNet-B0, and possibly an ensemble.
10. 3 random seeds, Grad-CAM, and a per-Fitzpatrick-type breakdown on SCIN.

**Start by summarising this handoff back to the user in a few lines, then present Step 4 and ask for approval.**
