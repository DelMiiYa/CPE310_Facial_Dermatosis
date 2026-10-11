# Handoff Prompt: Facial Dermatosis Classification (CPE310)

*Paste this entire file into a fresh AI assistant. It is self-contained. Read it fully before doing anything.*

---

## 1. Your role and working rules

You are an expert ML engineer continuing an **existing** PyTorch computer-vision project. Earlier assistants finished setup, the SCIN external test set, a manifest, source loaders, and a first gallery review. **Don't redo completed work and don't reverse approved decisions.**

**Strict rules:**
1. Work step by step. **Propose each step and wait for the user's approval before running it.** Never write the whole codebase at once.
2. **Never delete or modify anything in `data/raw/`.** Never use DermNet or clinical-only imagery for training.
3. Commit to the local git repo after each completed step. `data/` and `.env` are git-ignored. **Never print, log or commit `ROBOFLOW_API_KEY`.**
4. Stop at every **⏸ STOP** and let the user review.
5. Windows + PowerShell, Python 3.13. Don't use `cd`; run commands with the project root as the working directory. Prefer the package imports (`from src...`) over adding `sys.path` hacks in new files.
6. Read `agent_guidelines.md` first. Treat `project_prompt.md`, `download_data.py` and `filter_and_dedup.py` as **legacy**: the old pipeline pulled in DermNet, used exact-match hashing, and lost each image's source. Do not run them.

---

## 2. Project context

- **Goal:** a multi-class classifier for facial skin conditions from **smartphone photos** (full-face selfies *and* close-ups).
- **Classes:** `Acne`, `Rosacea`, `Dermatitis` (atopic eczema + contact dermatitis merged, as the literature survey suggests), `Normal`. Justify the merge in the final report.
- **Hardware:** RTX 3050 Laptop GPU, **4 GB VRAM**. Verified: `torch 2.6.0+cu124`, `mediapipe 0.10.35`, `timm`, `grad-cam`, `roboflow 1.7.1` are installed. Always use mixed precision (`torch.amp.autocast`), batch size 16–32, and DataLoaders with `pin_memory=True`, `persistent_workers=True`, `num_workers=4`.
- **Why the data rules are strict:** in v1, models trained on clinical DermNet photos got 95% accuracy on DermNet and **failed completely on real selfies** (domain shift). Training data must look like phone photos.
- **Literature survey** (`Literature_and_Papers/Literature Survey.pdf`, in Thai) says:
  - Most prior work is binary; the hard real-world task is **acne vs. rosacea**.
  - Use EfficientNet-B0 / ResNet-50 with transfer learning.
  - **Evaluate across Fitzpatrick skin types.**
  - Compare a single model vs. an ensemble.
  - **Patch-based** input (forehead/nose/cheeks) beats whole-face input.

### Approved architecture
One patch classifier plus a pretrained face detector (MediaPipe) that decides how to crop:
- **Face detected (selfie):** crop forehead, nose, left cheek, right cheek, chin and classify each patch.
- **No face (close-up):** tile into 224×224 patches.
- **Aggregate** patch scores into one diagnosis and generate a Grad-CAM heatmap.

---

## 3. Approved decisions

1. **SCIN is the external test set only.** It is never used for training, validation or tuning. 337 facial images (149 cases) are already downloaded to `data/raw/scin/dataset/images/` and registered as `role=external_test`.
2. **DermNet-derived data is excluded.** That includes `massive-skin-disease-balanced-dataset` and most of `skin-assessment-dataset`.
3. **`data/manifest/manifest.csv` is the single source of truth.** Images stay where they are; rows point to them. Every later step reads and updates the manifest instead of copying folders.
4. Split **70/15/15, stratified by class, grouped by `subject_id`**, with deduplication done **before** splitting.

---

## 4. Current repo state (verified)

**Git:** `main`, latest commits: `53f4695` (handoff update), `548bd15` (Roboflow rosacea link added), `b98438c` (`.env` ignored), `81cabee` (Step 4 code).

**Code:**
```
src/manifest.py                 # schema, load/append/validate, generate_image_id
src/sources/scin.py             # SCIN metadata parsing, CLASS_GROUPS
src/sources/{iga34,yolo_acne,skin_assessment,skin_defects}.py   # loaders
scripts/scin_fetch_metadata.py, scin_explore.py, scin_download_filtered.py, generate_galleries.py
```
`.env` holds `ROBOFLOW_API_KEY` (git-ignored).

**Manifest (6,809 rows, all files exist):**

| role | source | label | images | `subject_id` |
|---|---|---|---|---|
| external_test | scin | Acne 58 / Dermatitis 245 / Rosacea 34 | 337 | case id (149 cases) |
| train_pool | iga34 | Acne | 687 | person folder |
| train_pool | yolo_acne | Acne | 927 | **empty** |
| train_pool | skin_defects | Acne | 30 | person folder |
| train_pool | skin_assessment | Acne 1,185 / Dermatitis 1,192 / Normal 1,240 / Rosacea 1,211 | 4,828 | **empty** |

`review_status` is `pending` for all training rows and `accepted` for SCIN.

**Roboflow downloads: NOT DONE.** `data/raw/` has no Roboflow folder yet. Doing this is your first job (see Task 1).

---

## 5. Known issues and findings (read before touching data)

1. **`skin_assessment` is contaminated.** Filenames like `07rosacea...jpeg` and `03eczema...jpeg` are repackaged DermNet; many images are not faces. The user's gallery review confirmed non-face body photos in `skin_assessment` Dermatitis and Normal. **Do not train Rosacea or Dermatitis on it.**
   - `Normal` (1,240) has filenames like `0_0_anhu_0216.jpeg`, which look like a different face dataset (possibly a usable source of Normal faces).
   - `Acne` (1,185) mixes web-scraped images with clinical close-ups like `acne-closed-comedo-28.jpeg`.
   - **Decide per class after looking at samples with the user.** Don't keep or drop silently.
2. **Rosacea is the data bottleneck.** Almost no phone-style rosacea exists locally. Fix it with Task 1.
3. **Dermatitis also needs a phone-style training source.** The face-skin-diseases fork (Task 1b) has about 99 "eksim" (eczema) images.
4. **5,755 training images have no `subject_id`** (`skin_assessment`, `yolo_acne`). Since the split is grouped by `subject_id`, empty IDs would each count as their own person and leak the same person or near-duplicates across train and test. Fill them after dedup (Task 3): one cluster ID per near-duplicate group.
5. **`review_status` is not recorded** even though the user reviewed galleries. After the user's per-source/class verdict, update the manifest (`accepted`/`rejected`).
6. **`yolo_acne` ignores its bounding boxes** and isn't confirmed phone-style. Look at samples before trusting it.
7. **Non-face images in SCIN.** The user found body photos in the SCIN gallery too (SCIN's `HEAD_OR_NECK` includes neck-only shots). Step 5's face filter must also mark these.

---

## 6. Tasks (do them in this order; propose each and wait for approval)

### Task 1: Download the latest Roboflow Rosacea dataset (REQUIRED, not yet done)
Dataset: `https://universe.roboflow.com/skin-face-ixq7s/rosacea-n6xhv-jnjno/` → workspace `skin-face-ixq7s`, project `rosacea-n6xhv-jnjno`.

1. Create `scripts/roboflow_download.py`. Read `ROBOFLOW_API_KEY` from `.env` (e.g. `python-dotenv`, or parse the file). **Never print the key.**
2. Connect with the `roboflow` SDK: `Roboflow(api_key=...).workspace("skin-face-ixq7s").project("rosacea-n6xhv-jnjno")`.
3. **List the project's versions and use the latest one** (print version numbers, image counts and any preprocessing/augmentation it reports). Prefer a version with **no augmentation**; if every version is augmented, download the highest-numbered one and tell the user so, because augmented copies cause dedup and leakage problems.
4. Download it (COCO or folder format) to `data/raw/roboflow-rosacea/`. Make it re-runnable: skip if the folder already has files.
5. Count images and classes. Roboflow exports may include other classes (e.g. non-rosacea); keep a record of the original class names.
6. Write `src/sources/roboflow_rosacea.py` in the same style as the other loaders. Register images with `source=roboflow_rosacea`, `label=Rosacea` (only for images whose Roboflow class is actually rosacea), `role=train_pool`, `review_status=pending`, and use `generate_image_id`.
7. **⏸ STOP:** generate a gallery with `scripts/generate_galleries.py` and ask the user whether these look like phone photos or clinical/GAN images. GAN images were used in Paper 5; they are not real phone photos, so the user may want them excluded.
8. Commit.

### Task 1b: Download the face-skin-diseases fork (REQUIRED, not yet done)
The user forked `facial-skin-dataset/face-skin-diseases` into their own workspace: **`d3l-m11/face-skin-diseases-bbveb`** (498 raw images: about 100 rosacea, 99 "eksim"/eczema, 100 acne). It may have no generated version yet.
1. In `scripts/roboflow_download.py`, list versions for workspace `d3l-m11`, project `face-skin-diseases-bbveb`. If **no version exists**, tell the user to create one in the Roboflow web UI with **no preprocessing and no augmentation**, and wait. (Don't try to export augmented data.)
2. Download to `data/raw/roboflow-face-skin/`, then register with a loader. Map `rosacea` → `Rosacea`, `eksim` → `Dermatitis`, `acne` → `Acne`, keeping the original class name in `orig_label`.
3. Show a gallery and **⏸ STOP** for review. Commit.

### Task 2: Review verdicts and source decisions (⏸ STOP)
Show the user a summary table per source × class and ask for decisions on `skin_assessment` Normal/Acne, `yolo_acne`, and each Roboflow set (see Section 5). Record the results in `review_status` (`accepted`/`rejected`). Keep rejected images in the manifest; don't delete files.

### Task 3: Step 5, MediaPipe face detection and cropping
- `src/detection.py` and `scripts/run_face_detection.py`. Run MediaPipe over the manifest (including SCIN) and fill `face_detected`.
- For images with no face, the user (or a rule agreed with the user) decides: valid facial close-up vs. non-face body part (`rejected`).
- Implement patch extraction: forehead / nose / L cheek / R cheek / chin for selfies, 224×224 tiles for close-ups.
- Commit.

### Task 4: Step 6, near-duplicate removal
- `scripts/deduplicate.py` with `imagehash.phash`, **Hamming distance ≤ 6–8**, across **all sources**, **before splitting**. Store `phash` in the manifest.
- Flag near-duplicates that have **conflicting labels**; don't silently keep one.
- Assign a `subject_id` to every image that doesn't have one (cluster ID), as described in Section 5.
- Keep SCIN separate: any training image that near-matches an SCIN test image must be **dropped from training** to avoid leakage into the external test set.

### Task 5: Step 7, manual review (⏸ STOP)
Show sample patches and dedup results to the user.

### Task 6: Steps 8–9, split and ResNet-50 baseline
- 70/15/15, stratified by class, grouped by `subject_id`. SCIN stays external.
- ResNet-50 with class-weighted cross-entropy, mixed precision, **Macro-F1**, **3 seeds**.

### Task 7: Later (after the baseline)
EfficientNet-B0, a possible ensemble, **Grad-CAM** on all models (check the model isn't looking at backgrounds), and a **per-Fitzpatrick-type breakdown on SCIN** (the `fst` column).

### Housekeeping to fold into the steps above
- Update `requirements.txt` with the packages actually used (torch is installed with the cu124 index; mediapipe, timm, grad-cam, roboflow, python-dotenv).
- Fix the minor code smells when you touch the files (repeated `sys.path.append`).

---

**Start by briefly confirming you've read this, then propose Task 1 (the Roboflow Rosacea download) and wait for the user's approval.**
