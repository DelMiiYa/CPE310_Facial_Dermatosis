# Project Handoff Summary: CPE310 Facial Dermatosis Classifier

**Generated On:** 2026-10-18  
**Repository Working Directory:** `e:\CPE310FacialDermatorsisProject`  
**Primary Documentation:**
- [`project_prompt.md`](file:///e:/CPE310FacialDermatorsisProject/project_prompt.md) (Master 5-phase project guidelines & constraints)
- [`agent_guidelines.md`](file:///e:/CPE310FacialDermatorsisProject/agent_guidelines.md) (Lessons learned, domain gap prevention, anti-hallucination rules)
- [`HANDOFF_PROMPT.md`](file:///e:/CPE310FacialDermatorsisProject/HANDOFF_PROMPT.md) (Detailed step-by-step engineering handoff and approved decisions)
- [`Literature Survey.pdf`](file:///e:/CPE310FacialDermatorsisProject/Literature_and_Papers/Literature%20Survey.pdf) (Literature review on facial dermatosis & patch classification)

---

## 1. Executive Overview & Objectives

The goal of this project is to develop a robust, multi-class deep learning classifier for facial dermatological conditions intended to evaluate **real-world smartphone photos** (both full-face selfies and close-ups).

### Target Classes (4-Class Problem):
1. **Acne** (Acne Vulgaris)
2. **Rosacea**
3. **Dermatitis** (Atopic Eczema + Contact Dermatitis merged)
4. **Normal** (Healthy facial skin control)

### Core Architectural Decisions:
* **Patch-Based Classification + MediaPipe Face Detector:**
  - Full-face selfie input: MediaPipe detects the face and landmarks to crop specific facial regions (forehead, nose, left cheek, right cheek, chin).
  - Close-up input (no full face detected): Image is tiled into standard patches.
  - Aggregation: Patch predictions are aggregated into a single diagnosis with Grad-CAM visual heatmaps.
  - Rationale: Data is scarce; patches from both selfies and close-ups look visually similar, allowing one unified patch classifier to maximize data efficiency.
* **Hardware Profile & Optimizations:**
  - Local GPU: **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** running on Windows with Python 3.13 and PyTorch 2.6 (`+cu124`).
  - Mixed precision training (`torch.amp.autocast`) is mandatory.
  - Batch sizes: 16–32 to avoid Out-Of-Memory (OOM) exceptions.
  - DataLoaders: `pin_memory=True`, `persistent_workers=True`, `num_workers=4`.

---

## 2. Critical Lessons Learned & Approved Decisions

1. **The Domain Gap Rule (CRITICAL LESSON FROM V1):**
   - In v1, training on DermNet clinical macro-lens photos achieved 95% accuracy on clinical test sets but **completely failed (predicted "Normal") on real smartphone selfies**.
   - **Hard Rule:** All training data must visually match production smartphone photos. Clinical textbook photography (e.g. DermNet) is strictly prohibited from training.
2. **SCIN is Strictly the External Test Set:**
   - Google SCIN dataset cases are held out exclusively as an **unseen external test benchmark**. SCIN data is **never** used for training, validation, or hyperparameter tuning.
3. **Manifest Single Source of Truth (`data/manifest/manifest.csv`):**
   - All dataset metadata is tracked in a unified CSV manifest pointing to existing files. Raw images remain untouched in `data/raw/` (no file copying or duplicate folders).
4. **Legacy Deprecation:**
   - Previous prototype scripts `download_data.py`, `filter_and_dedup.py`, and the folder `data/processed/` are **deprecated legacy** and must not be used.

---

## 3. Work Completed to Date

### Step 4A: Manifest Infrastructure & SCIN External Test Set
* Implemented [`src/manifest.py`](file:///e:/CPE310FacialDermatorsisProject/src/manifest.py) with schema validation, uniqueness constraints, and safe append operations.
  - Schema: `image_id, source, subject_id, path, orig_label, label, label_confidence, body_part, shot_type, fst, role, phash, face_detected, review_status, split`.
* Implemented [`scripts/scin_download_filtered.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/scin_download_filtered.py):
  - Filtered SCIN cases for facial lesions (`body_parts_head_or_neck == YES`) for Acne, Rosacea, and Dermatitis with dermatologist weighted label $\ge 0.5$.
  - Downloaded **337 verified facial images** directly to `data/raw/scin/dataset/images/` tagged as `role=external_test`.

### Step 4B: Local Train Pool Inventory (Read-Only Loaders)
Implemented modular loaders under `src/sources/` registering raw local images into `data/manifest/manifest.csv` (`role=train_pool`):
* [`src/sources/iga34.py`](file:///e:/CPE310FacialDermatorsisProject/src/sources/iga34.py): 687 genuine smartphone selfies across 229 subjects (`subject_id = person_id`).
* [`src/sources/yolo_acne.py`](file:///e:/CPE310FacialDermatorsisProject/src/sources/yolo_acne.py): 927 acne images/crops.
* [`src/sources/skin_defects.py`](file:///e:/CPE310FacialDermatorsisProject/src/sources/skin_defects.py): 30 acne selfies across 10 subjects (front/left/right angles).
* [`src/sources/skin_assessment.py`](file:///e:/CPE310FacialDermatorsisProject/src/sources/skin_assessment.py): 4,828 candidate images across acne, Rosacea, Eczema, and normal.

### Step 4C: Visual Review Galleries & User Audit Findings
* Created [`scripts/generate_galleries.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/generate_galleries.py) generating HTML contact sheets for each source $\times$ class in `data/reports/galleries/`.
* **User Manual Audit Results:**
  1. **Label Accuracy:** Visual inspection confirmed class labels are accurate.
  2. **Non-Face Images Identified:** `scin - Dermatitis`, `skin_assessment - Dermatitis`, and `skin_assessment - Normal` contain non-facial body images (arms, legs, hands, torso). MediaPipe face filtering (Step 5) is required to eliminate them.
  3. **Critical Discovery on `skin_assessment`:** Filename examination revealed `skin_assessment` Rosacea and Dermatitis images are repackaged **DermNet** clinical photos (`07rosacea...`, `03eczema...`). Training on them would trigger the exact domain gap from v1.
  4. **The Rosacea Gap:** We have plenty of selfie data for Acne (~1,644 images), but zero verified smartphone selfies for Rosacea in the training pool.

### Sourcing & Roboflow Setup:
* Securely stored the user-provided Roboflow API key in `.env` (git-ignored, updated `.gitignore`).
* Installed `roboflow` SDK (1.7.1). Verified CUDA with PyTorch 2.6 on RTX 3050.
* User added dedicated Roboflow Rosacea dataset link to line 9 of [`datasets_links.txt`](file:///e:/CPE310FacialDermatorsisProject/datasets_links.txt):
  `https://universe.roboflow.com/skin-face-ixq7s/rosacea-n6xhv-jnjno/`
* In addition, project `facial-skin-diseases` was forked into the user's Roboflow workspace as `d3l-m11/face-skin-diseases-bbveb` (498 raw images).

---

## 4. Current Dataset Inventory Status

| Class | Total Manifest Count | Genuine Smartphone Count | Quality / Domain Assessment | Status |
| :--- | :---: | :---: | :--- | :--- |
| **Acne** | 2,829 | **~1,644** | `iga34` (687) + `yolo_acne` (927) + `skin_defects` (30) are genuine smartphone photos. | ✅ **Ready** |
| **Rosacea** | 1,211 | **0** | `skin_assessment` is 100% DermNet clinical imagery. Need real phone photos. | 🚨 **Target of Current Sourcing** |
| **Dermatitis** | 1,192 | **< 50** | `skin_assessment` is DermNet + body parts (arms/legs). | ⚠️ **Needs Filtering & Augmentation** |
| **Normal** | 1,240 | **~200** | Contains celebrity face crops and skin patches; easily augmented with FFHQ/selfies. | 🟡 **Manageable** |
| **External Test (SCIN)** | **337** | **337** | Real patient submissions (Acne: 58, Dermatitis: 245, Rosacea: 34). Locked holdout test set. | 🔒 **Locked** |

---

## 5. Next Immediate Implementation Steps

When resuming work, follow these sequential steps:

### Task 1: Ingest Roboflow Rosacea Dataset (Resolve the Rosacea Gap)
* Target dataset: `skin-face-ixq7s/rosacea-n6xhv-jnjno` from [`datasets_links.txt`](file:///e:/CPE310FacialDermatorsisProject/datasets_links.txt) (or user workspace `d3l-m11/face-skin-diseases-bbveb`).
* Download high-resolution / unaugmented images to `data/raw/roboflow-rosacea/` using `ROBOFLOW_API_KEY` from `.env`.
* Create `src/sources/roboflow_rosacea.py` to register images in `data/manifest/manifest.csv` as `label=Rosacea`, `role=train_pool`.
* Generate review thumbnails to verify domain match.

### Task 2: Step 5 — MediaPipe Face Detection & Patch Extraction
* Implement `src/detection.py` and `scripts/run_face_detection.py`:
  - Run MediaPipe face detector over all manifest images to populate `face_detected` boolean.
  - Automatically reject non-face images (arms, hands, legs) from the training pool.
  - For selfies (`face_detected == True`): crop landmark-based facial patches (forehead, left cheek, right cheek, nose, chin).
  - For close-ups: tile into $224 \times 224$ patches.

### Task 3: Step 6 — Perceptual Hash Deduplication
* Implement `scripts/deduplicate.py` using `imagehash.phash`.
* Deduplicate across ALL sources with Hamming distance threshold $\le 6\text{--}8$.
* **Critical Rule:** Deduplication must run **before** dataset splitting to prevent data leakage between train, val, and test.

### Task 4: Step 7 — Manual Review Pause (⏸ STOP Point)
* Present sample cropped patches and deduplication metrics to the user for review and approval.

### Task 5: Step 8 & 9 — Stratified Split & ResNet-50 Baseline
* Execute 70/15/15 train/val/test split grouped by `subject_id` to ensure no person's photos cross split boundaries. SCIN remains pure external test.
* Train ResNet-50 baseline using class-weighted cross-entropy and mixed precision (`torch.amp.autocast`), evaluated on Macro-F1 across 3 random seeds with Grad-CAM explainability.

---

## 6. Repository File & Directory Map

```
E:\CPE310FACIALDERMATORSISPROJECT
├── .env                                # Private API keys (git-ignored)
├── .gitignore                          # Ignores data/, .env, runs/, checkpoints/
├── agent_guidelines.md                 # Core engineering rules & domain shift guidelines
├── project_prompt.md                   # 5-phase master project instructions
├── HANDOFF_PROMPT.md                   # Complete assistant-to-assistant handoff prompt
├── Project_handoff_summary.md          # This up-to-date summary document
├── datasets_links.txt                  # Origin URLs for raw datasets & Roboflow
├── requirements.txt                    # Project python dependencies
├── data/
│   ├── manifest/
│   │   └── manifest.csv                # Single source of truth (6,809 registered images)
│   ├── raw/                            # Untouched raw dataset directories
│   │   ├── acne-dataset-in-yolov8-format/
│   │   ├── scin/dataset/images/        # 337 downloaded SCIN external test images
│   │   ├── skin-assessment-dataset/
│   │   ├── skin-defects-acne-redness-and-bags-under-the-eyes/
│   │   └── skin-problems-34-on-the-iga-scale/
│   └── reports/
│       ├── scin_explore.md             # SCIN exploratory analysis report
│       └── galleries/                  # Generated HTML review contact sheets
├── scripts/
│   ├── scin_fetch_metadata.py          # Downloads SCIN cases & labels metadata
│   ├── scin_explore.py                 # Generates SCIN exploratory analysis report
│   ├── scin_download_filtered.py       # Filters & downloads SCIN facial test set
│   └── generate_galleries.py           # Generates HTML review galleries
├── src/
│   ├── manifest.py                     # Manifest schema, validation & append helpers
│   └── sources/
│       ├── scin.py                     # SCIN data parser & label grouping
│       ├── iga34.py                    # IGA34 acne selfie loader
│       ├── yolo_acne.py                # YOLO acne dataset loader
│       ├── skin_assessment.py          # Skin assessment dataset loader
│       └── skin_defects.py             # Skin defects selfie loader
└── Literature_and_Papers/
    └── Literature Survey.pdf           # Literature survey & comparative analysis
```
