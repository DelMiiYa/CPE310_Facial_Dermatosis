# Handoff Summary: Facial Dermatosis Classification (CPE310)

> **To the Next AI Assistant:**  
> Read this document completely before taking any action. It summarizes the current state, completed work, verified decisions, and immediate next steps. Work step by step, propose each step, and wait for user approval before executing code.

---

## 1. Operating Rules & Constraints (STRICT)

1. **Step-by-step with approval:** Propose one task at a time and wait for user confirmation. Never generate large parts of the codebase at once.
2. **Never touch `data/raw/`:** Never delete or alter existing raw files.
3. **Keep secrets secure:** `.env` and `data/` are git-ignored. **NEVER print, log, or commit `ROBOFLOW_API_KEY`.**
4. **Single Source of Truth:** `data/manifest/manifest.csv` tracks all images, their roles, paths, labels, and review statuses. Images remain in place; the manifest references them.
5. **Domain Alignment Rule (Crucial):** Prior iteration failed because clinical photos (DermNet) caused severe domain shift when predicting real smartphone selfies. We train ONLY on real-world phone photos/selfies, not clinical textbook macro photography.
6. **Hardware Limits:** NVIDIA RTX 3050 Laptop GPU (**4 GB VRAM**), Windows 11 PowerShell, Python 3.13. Always use mixed precision (`torch.amp.autocast`), batch size 16–32, and DataLoaders with `pin_memory=True`, `persistent_workers=True`, `num_workers=4`.
7. **Git:** Commit after each completed step.

---

## 2. Current Repository State (Verified)

- **Branch:** `main`
- **Latest Commit:** `a51b601` (`Step 4b / Task 1: Add Roboflow rosacea download and loader`)
- **Key Modules & Scripts:**
  - `src/manifest.py`: Schema definition, loading, validating, appending, and ID generation (`generate_image_id`).
  - `src/sources/scin.py`: External test set parser.
  - `src/sources/{iga34, yolo_acne, skin_assessment, skin_defects}.py`: Source loaders.
  - `src/sources/roboflow_rosacea.py`: **NEW** loader for the downloaded Roboflow rosacea dataset.
  - `scripts/roboflow_download.py`: **NEW** download utility using the Roboflow SDK.
  - `scripts/generate_galleries.py`: Generates HTML review galleries in `data/reports/galleries/`.

---

## 3. Dataset & Manifest Breakdown

**Total Rows in `data/manifest/manifest.csv`:** **7,897**

| Role | Source | Label | Images | Notes |
|---|---|---|---|---|
| `external_test` | `scin` | Acne (58) / Dermatitis (245) / Rosacea (34) | **337** | Google SCIN external test set (149 cases). **Never used for training or tuning.** |
| `train_pool` | `iga34` | Acne | **687** | High-quality facial selfie photos. `subject_id` populated. |
| `train_pool` | `roboflow_rosacea` | Rosacea | **1,088** | **Downloaded & registered in Task 1.** Approved by user after gallery inspection. Bounding box class `redness` (`orig_label: "redness"`). |
| `train_pool` | `yolo_acne` | Acne | **927** | Bounding box dataset, pending domain review. `subject_id` empty. |
| `train_pool` | `skin_defects` | Acne | **30** | Small batch with person folder IDs. |
| `train_pool` | `skin_assessment` | Acne (1,185) / Dermatitis (1,192) / Normal (1,240) / Rosacea (1,211) | **4,828** | **Contaminated with DermNet & non-face shots.** User verified non-face photos in Dermatitis/Normal. Rosacea & Dermatitis to be rejected; Normal to be evaluated. |

---

## 4. What Was Just Completed (Task 1)

1. Identified that user forked `skin-face-ixq7s/rosacea-n6xhv-jnjno` into their workspace: `d3l-m11/rosacea-n6xhv-jnjno-rruul`.
2. Initialized Version 1 with **no preprocessing and no augmentation** via the Roboflow API.
3. Downloaded the dataset in COCO format to `data/raw/roboflow-rosacea/` (1,116 images).
4. Implemented `src/sources/roboflow_rosacea.py`, filtering 1,088 images annotated with `redness` into the manifest with `label="Rosacea"`.
5. Generated the HTML review gallery `data/reports/galleries/roboflow_rosacea_Rosacea.html`.
6. User visually inspected the gallery and approved it: *"looks good to me"*.
7. Committed code in `a51b601`.

---

## 5. Immediate Next Steps (Follow in exact order)

### ⏩ Next Immediate Action: Task 1b (Download face-skin-diseases fork)

The user has forked `facial-skin-dataset/face-skin-diseases` into their Roboflow workspace: **`d3l-m11/face-skin-diseases-bbveb`**.
- Contains 498 raw images: ~100 rosacea, 99 "eksim" (eczema/dermatitis), 100 acne, 100 herpes, 99 panu.
- This will provide the critical phone-style training data for **Dermatitis** (`eksim`)!

**Procedure for Task 1b:**
1. Check project versions for `d3l-m11/face-skin-diseases-bbveb`. If 0 versions exist, initialize version generation with `{"preprocessing": {}, "augmentation": {}}` via Roboflow API (or ask the user to click Generate in the Roboflow web UI with no augmentations).
2. Download to `data/raw/roboflow-face-skin/`.
3. Create loader `src/sources/roboflow_face_skin.py`. Map:
   - `eksim` → `label="Dermatitis"`, `orig_label="eksim"`
   - `rosacea` → `label="Rosacea"`, `orig_label="rosacea"`
   - `acne` → `label="Acne"`, `orig_label="acne"`
   - (Skip `herpes` and `panu` as they are outside our 4 target classes).
   - Set `role="train_pool"`, `review_status="pending"`, and generate unique IDs.
4. Run `scripts/generate_galleries.py` and **⏸ STOP** for the user to review the gallery.
5. Commit.

---

### Subsequent Tasks:

- **Task 2: Review Verdicts & Source Decisions (⏸ STOP):**
  - Present source × class table.
  - Formally record `review_status="rejected"` on `skin_assessment` Rosacea/Dermatitis. Decide on `skin_assessment` Normal and `yolo_acne`.
- **Task 3: Step 5 — MediaPipe Face Detection & Patch Extraction:**
  - Create `src/detection.py` and `scripts/run_face_detection.py`.
  - Mark `face_detected` boolean across all manifest rows (including SCIN).
  - Crop 5 facial patches (forehead, nose, left cheek, right cheek, chin) for selfie images, and 224×224 tiling for close-ups.
- **Task 4: Step 6 — Perceptual Deduplication:**
  - `scripts/deduplicate.py` using `imagehash.phash` (Hamming distance ≤ 6–8) across all sources.
  - Drop any training images that match SCIN external test images to prevent leakage.
  - Assign cluster `subject_id` to images currently lacking subject IDs.
- **Task 5: Step 7 — Manual Review of Patches & Dedup (⏸ STOP).**
- **Task 6: Steps 8–9 — Train/Val/Test Split & ResNet-50 Baseline:**
  - 70/15/15 split grouped by `subject_id`, stratified by class.
  - ResNet-50 baseline with class-weighted CE, mixed precision, Macro-F1 across 3 random seeds.
- **Task 7: Model Expansion & Final Evaluation:**
  - EfficientNet-B0, model ensemble, Grad-CAM interpretability, and Fitzpatrick Skin Type breakdown on SCIN (`fst` column).

---

## 6. How to Run Key Commands (PowerShell from project root)

```powershell
# Activate Python environment & check manifest summary
python -c "from src.manifest import load_manifest; df = load_manifest(); print(df.groupby(['role', 'source', 'label']).size())"

# Generate / update HTML review galleries
python scripts/generate_galleries.py

# Run git status
git status
```
