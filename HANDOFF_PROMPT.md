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

**Git:** `main`, check `git log` for the current head. Relevant commits: `a51b601` (Roboflow rosacea download + loader), `33a3e54` (this handoff), `81cabee` (Step 4 code). `handoff_summary.md` / `HANDOFF_SUMMARY.md` is a previous agent's summary; **this file is the authority** where they disagree.

**Code:**
```
src/manifest.py                 # schema, load/append/validate, generate_image_id
src/sources/scin.py             # SCIN metadata parsing, CLASS_GROUPS
src/sources/{iga34,yolo_acne,skin_assessment,skin_defects,roboflow_rosacea}.py   # loaders
scripts/scin_fetch_metadata.py, scin_explore.py, scin_download_filtered.py, generate_galleries.py, roboflow_download.py
```
`.env` holds `ROBOFLOW_API_KEY` (git-ignored).

**Manifest Status (8,196 rows total, all files exist on disk):**

| Role | Source | Label | Accepted (Clean) | Rejected (Preserved) | Total | Notes |
|---|---|---|---|---|---|---|
| `external_test` | `scin` | Acne (58) / Rosacea (34) / Dermatitis (245) | **337** | 0 | 337 | Strictly held-out test set (149 cases) |
| `train_pool` | `iga34` | Acne | **686** | 1 | 687 | Real phone selfies (229 subjects) |
| `train_pool` | `skin_defects` | Acne | **27** | 3 | 30 | Small selfie batch (10 subjects) |
| `train_pool` | `roboflow_face_skin` | Acne (100) / Rosacea (48) / Dermatitis (93) | **241** | 58 | 299 | Distinct phone images (58 duplicates/conflicts rejected) |
| `train_pool` | `roboflow_rosacea` | Rosacea | **616** | 472 | 1,088 | Unique original photos (444 augmented duplicates rejected) |
| `train_pool` | `skin_assessment` | Normal | **644** | 596 | 1,240 | Verified face-detected normal shots (no-face rejected) |
| `train_pool` | `skin_assessment` | Acne (1,185) / Rosacea (1,211) / Dermatitis (1,192) | **0** | 3,588 | 3,588 | Rejected (DermNet / non-face / surplus) |
| `train_pool` | `yolo_acne` | Acne | **0** | 927 | 927 | Rejected (surplus augmented copies) |
| **Total** | | | **2,551** | **5,645** | **8,196** | **2,214 train_pool + 337 external_test** |

**Training pool distribution:** Acne: **813**, Rosacea: **664**, Normal: **644**, Dermatitis: **93**. All accepted rows have unified `subject_id`s. Zero test leakage with SCIN. Zero files deleted.

---

## 5. Known issues and findings (read before touching data)

1. **`skin_assessment` is contaminated.** Filenames like `07rosacea...jpeg` and `03eczema...jpeg` are repackaged DermNet; many images are not faces. The user's gallery review confirmed non-face body photos in `skin_assessment` Dermatitis and Normal. **Do not train Rosacea or Dermatitis on it.**
   - `Normal` (1,240) has filenames like `0_0_anhu_0216.jpeg`, which look like a different face dataset (possibly a usable source of Normal faces).
   - `Acne` (1,185) mixes web-scraped images with clinical close-ups like `acne-closed-comedo-28.jpeg`.
   - **Decide per class after looking at samples with the user.** Don't keep or drop silently.
2. **Rosacea is the data bottleneck.** Almost no phone-style rosacea exists locally. Fix it with Task 1.
3. **Dermatitis also needs a phone-style training source.** The face-skin-diseases fork (Task 1b) has about 99 "eksim" (eczema) images.
4. **5,755 training images have no `subject_id`** (`skin_assessment`, `yolo_acne`). Since the split is grouped by `subject_id`, empty IDs would each count as their own person and leak the same person or near-duplicates across train and test. Fill them after dedup (Task 4): one cluster ID per near-duplicate group.
5. **`review_status` is not recorded** even though the user reviewed galleries. After the user's per-source/class verdict, update the manifest (`accepted`/`rejected`).
6. **`yolo_acne` ignores its bounding boxes** and isn't confirmed phone-style. Look at samples before trusting it.
7. **Non-face images in SCIN.** The user found body photos in the SCIN gallery too (SCIN's `HEAD_OR_NECK` includes neck-only shots). Step 5's face filter must also mark these.
8. **`roboflow_rosacea` is NOT unaugmented, despite the "no augmentation" export setting.** Files are named `<original>.rf.<hash>.jpg` (Roboflow's naming for augmented variants), with up to 10 copies of the same original in train. The 1,116 downloaded images come from only **669 unique originals** (train 506, valid 161, test 99), and the same original appears across Roboflow's own splits (57 train/valid, 28 train/test, 23 valid/test). The augmentation was most likely baked into the images in the user's fork. Consequences: (a) the real number of distinct rosacea photos is about 700 or fewer, not 1,088; (b) pHash dedup is mandatory before any split (Task 4); (c) never use Roboflow's train/valid/test folders as splits. Apply the same check to every new Roboflow download.
9. **`roboflow_rosacea` labels are only a `redness` bounding box** (the annotation classes are `acne-GJuB` and `redness`), not a rosacea diagnosis. Any photo with redness gets `label=Rosacea`. Some filenames (`Gul-Hastaligi-...`, `kering-48-...`, `IMG-...-WA...`) suggest a mix of web and phone images. The user's approval was based on a gallery sample of about 60 images, so also review more images after dedup, and ask the user if any clinical-looking or GAN-looking images should be excluded.
10. **Class balance (after dedup this will shift):** training currently has Acne about 2,800, Rosacea (Roboflow) about 700 distinct, Dermatitis 0 (until Task 1b), Normal undecided. Use class-weighted cross-entropy and report Macro-F1.

---

## 6. Tasks (do them in this order; propose each and wait for approval)

### Task 1: Roboflow Rosacea download ✅ DONE (commit `a51b601`), with caveats
Done: `scripts/roboflow_download.py` (key read from `.env`, never printed) and `src/sources/roboflow_rosacea.py`. The fork `d3l-m11/rosacea-n6xhv-jnjno-rruul` was exported as version 1 (COCO) to `data/raw/roboflow-rosacea/` (1,116 images; 1,088 with a `redness` box registered as `label=Rosacea`, `orig_label=redness`). The user said the gallery sample "looks good".
**Do not redo it. Caveats are in Known issues 8 and 9** (the set is full of augmented copies, and the label is only a "redness" box). If the user wants real originals, `skin-face-ixq7s/rosacea-n6xhv-jnjno` (the source project, not the fork) can be checked for an unaugmented version.

### Task 1b: face-skin-diseases fork ✅ DONE (commit `970fa5a`), user reviewed galleries: "looks good"
`d3l-m11/face-skin-diseases-bbveb` Version 1 (generated via API, no preprocessing/augmentation) → `data/raw/roboflow-face-skin/`. All 498 files are distinct originals (no `.rf.` duplicates, no train/test overlap). Registered 299 images as `source=roboflow_face_skin`, `role=train_pool`: Acne 100 (`acne`), Dermatitis 99 (`eksim`), Rosacea 100 (`rosacea`). Skipped `herpes` (100) and `panu` (99). Loader: `src/sources/roboflow_face_skin.py`. Manifest is now 8,196 rows. Do not redo.

### Task 2: Review verdicts and source decisions (⏸ STOP), PARTLY DECIDED
Decided with the user so far:
- Roboflow face-skin set: **accepted** as good phone-style data (record `review_status=accepted` for those 299 rows if not yet set).
- Recommended and to be confirmed: **reject `skin_assessment` Rosacea (1,211) and Dermatitis (1,192)** (repackaged DermNet / non-face). Record `rejected`; keep files.

**Deferred until after Task 3 (face detection), so the user only judges real faces:**
- `skin_assessment` Normal (1,240) and Acne (1,185), `yolo_acne` (927): show galleries of face-detected images, then ask the user to keep/reject.
- **Normal source (biggest gap, no source decided).** Options: Normal faces from `skin_assessment` after the face filter; CelebA (as in Paper 5; celebrity photos, not phone selfies); or photos the user collects with consent. Ask the user.
- Dermatitis is thin (99 training images, plus SCIN is test-only), so use class-weighted loss and tell the user about it.
### Task 3: Step 5, MediaPipe face detection and cropping ✅ DONE (commit `0583a56`)
MediaPipe Tasks Vision `blaze_face_short_range.tflite` + `face_landmarker.task` integrated in `src/detection.py` and run via `scripts/run_face_detection.py`.
- 8,196 images evaluated, zero failures.
- Manifest schema updated with `face_detected`, `face_conf`, `face_area_frac`.
- 2,827 face-detected images (34.5%); 5,369 close-ups / non-face images (65.5%).
- 24 sample patches verified in `data/reports/galleries/step5_patch_samples.html`.
- Full detection summary report saved at `data/reports/step5_face_detection_summary.md`.

### Task 4: Step 6, near-duplicate removal & subject clustering ✅ DONE (script `scripts/deduplicate.py`)
- Computed 64-bit canonical and horizontal-flip pHashes for all 8,196 images.
- Applied Task 2 source filtering: `skin_assessment` Rosacea/Dermatitis/Acne rejected (DermNet/non-face); `skin_assessment Normal` no-face rejected (512), faces accepted (644 distinct after dedup); `yolo_acne` surplus rejected (927).
- SCIN external test leakage check: **0 leaks** found (zero overlap between training candidates and 337 SCIN test images).
- `roboflow_rosacea`: collapsed from 1,088 images down to **616 distinct accepted originals** (444 redundant augmented copies rejected + 28 cross-source/conflict duplicates rejected).
- Cross-dataset conflict check: detected and rejected **3 label conflict clusters** across datasets (e.g. conflicting Acne/Rosacea or Rosacea/Dermatitis).
- Unified `subject_id`s assigned across all clusters and subjects (no empty `subject_id`s remain in the training pool).
- Full summary report saved at `data/reports/step6_deduplication_summary.md`.

### Task 5: Step 7, manual review ✅ DONE
- Clean training pool reviewed: **2,214 images** (Acne: 813 / 330 subjects, Rosacea: 664 / 664 subjects, Normal: 644 / 404 subjects, Dermatitis: 93 / 93 subjects).
- SCIN external test set: **337 images** (Acne: 58, Rosacea: 34, Dermatitis: 245).
- Patch extraction preview verified: `data/reports/galleries/step5_patch_samples.html`.

### Task 6: Steps 8–9, split and ResNet-50 baseline
- **Step 8: 70/15/15 stratified grouped split ✅ DONE (script `scripts/split_dataset.py`)**
  - Stratified by class and grouped by `subject_id` (zero subject leakage across train/val/test).
  - Train: 1,617 images (1,047 subjects) - Acne 591, Rosacea 466, Normal 497, Dermatitis 63.
  - Val: 294 images (222 subjects) - Acne 109, Rosacea 99, Normal 71, Dermatitis 15.
  - In-Domain Test: 303 images (222 subjects) - Acne 113, Rosacea 99, Normal 76, Dermatitis 15.
  - External Test (SCIN): 337 images (149 cases) - strictly held-out.
  - Report saved to `data/reports/step8_split_summary.md`. Manifest updated.
- **Step 9: ResNet-50 baseline (CURRENT STEP)**
  - PyTorch Dataset (`src/dataset.py`) supporting patch-based training & inference.
  - Model & Loss (`src/model.py`): ResNet-50 with inverse-frequency class weights ($w_c = \frac{N_{total}}{C \times N_c}$).
  - Training pipeline (`scripts/train_baseline.py`): Mixed precision (`torch.amp.autocast`), AdamW, CosineAnnealingLR, batch size 32, pin_memory=True, persistent_workers=True, num_workers=4.
  - Metrics tracked: Macro-F1 (primary), per-class F1, Accuracy, Loss across 3 seeds. Best checkpoint to `checkpoints/resnet50_baseline_best.pt`.
  - In-domain test and strictly held-out SCIN external test evaluation.

### Task 7: Later (after the baseline)
EfficientNet-B0, model ensemble comparison, **Grad-CAM** interpretability verification, and **per-Fitzpatrick-type breakdown on SCIN** (`fst` column).

### Housekeeping to fold into the steps above
- Update `requirements.txt` with the packages actually used (torch is installed with the cu124 index; mediapipe, timm, grad-cam, roboflow, python-dotenv).
- Fix the minor code smells when you touch the files (repeated `sys.path.append`).

---

**Current status:** Step 8 (Dataset Split) is complete. Next is implementing Step 9: PyTorch Dataset, Model architecture, and Training Pipeline for ResNet-50 baseline.
