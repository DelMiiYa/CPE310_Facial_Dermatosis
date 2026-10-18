# Project Handoff Summary: CPE310 Facial Dermatosis Classifier

**Last Updated:** 2026-10-18  
**Repository Working Directory:** `e:\CPE310FacialDermatorsisProject`  
**Primary Documentation:**
- [`HANDOFF_PROMPT.md`](file:///e:/CPE310FacialDermatorsisProject/HANDOFF_PROMPT.md) (Authoritative engineering handoff document)
- [`agent_guidelines.md`](file:///e:/CPE310FacialDermatorsisProject/agent_guidelines.md) (Core rules, domain gap avoidance, low-VRAM constraints)
- [`project_prompt.md`](file:///e:/CPE310FacialDermatorsisProject/project_prompt.md) (High-level 5-phase project guidelines)
- [`Literature Survey.pdf`](file:///e:/CPE310FacialDermatorsisProject/Literature_and_Papers/Literature%20Survey.pdf) (Clinical rationale and visual overlap of facial conditions)

---

## 1. Executive Overview & Problem Definition

The objective is to train and evaluate a robust, multi-class deep learning classifier for facial skin diseases capable of diagnosing **real-world smartphone imagery** (both full-face selfies and macro close-ups).

### Target Classes (4-Class Problem):
1. **Acne** (Acne Vulgaris)
2. **Rosacea**
3. **Dermatitis** (Atopic Eczema + Contact Dermatitis merged)
4. **Normal** (Healthy facial skin control)

### Core Architectural Decisions:
* **Patch-Based Classifier + MediaPipe Face Detector:**
  - **Full-face selfies (Face Detected):** MediaPipe tasks vision model detects the face and anatomical landmarks to extract 5 standardized skin crops: **forehead, nose, left cheek, right cheek, and chin**.
  - **Macro close-ups (No Face Detected):** The image is tiled into standard $224 \times 224$ patches.
  - **Inference Aggregation:** Predictions from all patches are aggregated via mean-probability pooling to produce a case-level diagnosis and Grad-CAM heatmap.
  - **Rationale:** Smartphone photos vary widely between zoomed-in lesions and full selfies. Decomposing both into skin patches standardizes the input domain and maximizes training data efficiency.
* **Hardware Profile & Optimizations:**
  - Hardware: **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** on Windows with Python 3.13 and PyTorch 2.6 (`+cu124`).
  - Mixed precision training (`torch.amp.autocast('cuda')`) is strictly enforced.
  - VRAM footprint maintained between **1.2 GB and 3.1 GB** with batch size 32.
  - DataLoaders configured with `pin_memory=True`, `persistent_workers=True`, and `num_workers=4`.

---

## 2. Hard Engineering Rules & Lessons Learned

1. **The Domain Gap Rule (Lesson from v1):**
   - In v1, training on DermNet clinical macro-lens photos achieved 95% test accuracy on clinical sets but **failed completely (predicted "Normal") on real user selfies**.
   - **Rule:** Training data must strictly represent smartphone camera capture. Textbook clinical imagery (DermNet) is strictly excluded from training.
2. **SCIN is Strictly the External Test Set:**
   - Google SCIN dataset (crowd-sourced smartphone photos) is held out exclusively as an **unseen external benchmark** (337 images / 149 cases). SCIN data is **never** used for training, validation, or hyperparameter tuning.
3. **Unified Manifest as Single Source of Truth (`data/manifest/manifest.csv`):**
   - Raw images remain completely untouched in `data/raw/`. All metadata, crops, hashes, labels, and split assignments are tracked in the manifest.
4. **Legacy Script Deprecation:**
   - Old prototype scripts (`download_data.py`, `filter_and_dedup.py`) and folder `data/processed/` are **deprecated legacy** and must not be used.

---

## 3. Work Completed to Date (Phases 1–4)

### Step 4A & 4B: Dataset Ingestion & Sourcing (Commits `81cabee`, `a51b601`, `970fa5a`)
* **Unified Manifest:** Created [`src/manifest.py`](file:///e:/CPE310FacialDermatorsisProject/src/manifest.py) schema (15 columns).
* **SCIN External Test Set:** Created [`scripts/scin_download_filtered.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/scin_download_filtered.py) to download 337 qualifying facial images (weighted label $\ge 0.5$) to `data/raw/scin/dataset/images/`.
* **Local Train Sourcing:**
  - `iga34`: 686 genuine acne selfies across 229 subjects.
  - `skin_defects`: 27 acne selfies across 10 subjects.
  - `roboflow_rosacea`: 1,116 images downloaded; collapsed down to **616 distinct original photos** (444 augmented duplicates rejected).
  - `roboflow_face_skin`: 241 distinct phone images (Acne: 100, Rosacea: 48, Dermatitis: 93).
  - `skin_assessment`: Inspected and found to be repackaged DermNet (`07rosacea...`, `03eczema...`). **Acne, Rosacea, and Dermatitis were rejected** to prevent domain contamination. Only **644 verified Normal faces** were accepted.
  - `yolo_acne`: 927 surplus augmented copies rejected.

### Step 5: MediaPipe Face Detection & Patch Extraction (Commit `0583a56`)
* Implemented [`src/detection.py`](file:///e:/CPE310FacialDermatorsisProject/src/detection.py) and [`scripts/run_face_detection.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/run_face_detection.py) using MediaPipe Tasks Vision (`blaze_face_short_range.tflite` + `face_landmarker.task`).
* Evaluated all 8,196 manifest images: 2,827 face-detected (34.5%), 5,369 close-ups/non-face (65.5%).
* Pre-extracted **14,330 patches** ($224 \times 224$) to `data/processed/patches/` with [`data/manifest/patches_manifest.csv`](file:///e:/CPE310FacialDermatorsisProject/data/manifest/patches_manifest.csv).
* Report: [`data/reports/step5_face_detection_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step5_face_detection_summary.md).

### Step 6: Perceptual Deduplication & Leakage Prevention (Commit `cf4d50c`)
* Implemented [`scripts/deduplicate.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/deduplicate.py) using 64-bit canonical & horizontally-flipped perceptual hashes (`imagehash.phash`, Hamming distance $\le 6\text{--}8$).
* **SCIN External Test Leakage Check:** **0 leaks** found (zero overlap between training pool and SCIN test set).
* Resolved 3 cross-dataset label conflicts and assigned unified `subject_id`s across all images.
* Report: [`data/reports/step6_deduplication_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step6_deduplication_summary.md).

### Step 7: Manual Review & Training Pool Audit
* Approved clean curated training pool of **2,214 images** across 1,491 subjects:
  - **Acne:** 813 images (330 subjects)
  - **Rosacea:** 664 images (664 subjects)
  - **Normal:** 644 images (404 subjects)
  - **Dermatitis:** 93 images (93 subjects)

### Step 8: Grouped Stratified 70/15/15 Split (Commit `6456e99`)
* Implemented [`scripts/split_dataset.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/split_dataset.py):
  - Stratified by class and **strictly grouped by `subject_id`** (zero subject leakage across splits).
  - **Train Split (70%):** 1,617 images / 1,047 subjects (Acne: 591, Rosacea: 466, Normal: 497, Dermatitis: 63)
  - **Validation Split (15%):** 294 images / 222 subjects (Acne: 109, Rosacea: 99, Normal: 71, Dermatitis: 15)
  - **In-Domain Test Split (15%):** 303 images / 222 subjects (Acne: 113, Rosacea: 99, Normal: 76, Dermatitis: 15)
  - **SCIN External Test Set:** 337 images / 149 cases (Acne: 58, Rosacea: 34, Dermatitis: 245)
* Report: [`data/reports/step8_split_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step8_split_summary.md).

### Step 9: ResNet-50 Baseline Training & Evaluation (Commit `5af89f7`)
* Built modular PyTorch training pipeline:
  - Dataset loader: [`src/dataset.py`](file:///e:/CPE310FacialDermatorsisProject/src/dataset.py)
  - Model wrapper: [`src/model.py`](file:///e:/CPE310FacialDermatorsisProject/src/model.py)
  - Training script: [`scripts/train_baseline.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/train_baseline.py)
  - Full inference pipeline: [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py)
* **Optimization Setup:** Pretrained ImageNet-1k `resnet50`, inverse-frequency class-weighted Cross-Entropy loss ($w_c = [0.6945, 0.8415, 5.0731, 0.8513]$), AdamW optimizer, cosine annealing schedule, mixed precision AMP.
* **Best Model Checkpoint:** `checkpoints/resnet50_baseline_best.pt` (Epoch 11).
* **Performance Summary:**
  - **In-Domain Test (303 images):**
    - **Macro-F1:** **0.8841** | **Overall Accuracy:** **92.74%**
    - Per-Class F1: Acne **0.9412**, Rosacea **0.9091**, Dermatitis **0.7059** (12/15 detected), Normal **0.9804** (75/76 detected).
  - **SCIN Held-Out External Test (337 images):**
    - Macro-F1: 0.1544 | Overall Accuracy: 21.66%
    - Acne Recall: **91.38%** (53/58 detected).
    - Dermatitis Recall: 5.31% (heavily confused as Acne due to crowd-sourced close-up domain gap).
    - Fitzpatrick Skin Type Breakdown: FST4 (31.6%) and FST5 (36.0%) accuracy exceeded lighter tones.
* Report: [`data/reports/step9_resnet50_baseline_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step9_resnet50_baseline_report.md).

---

## 4. Current Project State & Immediate Next Steps (Task 7)

> [!IMPORTANT]
> **Current Milestone:** Step 9 (ResNet-50 Baseline) is fully completed and verified. The codebase is clean on git commit `5af89f7` (and latest documentation commits).

### Upcoming Work (Task 7: Advanced Architectures, Ensemble & Explainability):
1. **EfficientNet-B0 Model:** Train EfficientNet-B0 on identical data folds, patches, and class weights to evaluate efficiency vs. ResNet-50.
2. **Model Ensemble:** Implement a weighted voting / probability average ensemble combining ResNet-50 and EfficientNet-B0 (fulfilling literature survey requirements).
3. **Grad-CAM Interpretability:** Generate visual heatmaps across both anatomical face patches and close-up tiles to verify that models focus on genuine lesion pathology.
4. **SCIN Domain Adaptation & Error Analysis:** Investigate strategies to improve Dermatitis sensitivity on crowd-sourced close-up photography.

---

## 5. Repository Structure Map

```
E:\CPE310FACIALDERMATORSISPROJECT
├── .env                                # Private API keys (git-ignored)
├── .gitignore                          # Ignores data/, .env, runs/, checkpoints/
├── agent_guidelines.md                 # Core engineering rules & domain shift guidelines
├── project_prompt.md                   # Master 5-phase project guidelines
├── HANDOFF_PROMPT.md                   # Authoritative full engineering handoff document
├── Project_handoff_summary.md          # Comprehensive executive summary (this document)
├── datasets_links.txt                  # Origin URLs for raw datasets & Roboflow
├── requirements.txt                    # Project python dependencies
├── checkpoints/
│   └── resnet50_baseline_best.pt       # Trained ResNet-50 baseline checkpoint
├── data/
│   ├── manifest/
│   │   ├── manifest.csv                # Master image manifest (8,196 rows)
│   │   └── patches_manifest.csv        # Extracted patches manifest (14,330 patches)
│   ├── processed/
│   │   └── patches/                    # Pre-extracted 224x224 training patches
│   ├── raw/                            # Untouched raw dataset folders
│   │   ├── acne-dataset-in-yolov8-format/
│   │   ├── roboflow-face-skin/
│   │   ├── roboflow-rosacea/
│   │   ├── scin/dataset/images/        # 337 SCIN external test images
│   │   ├── skin-assessment-dataset/
│   │   ├── skin-defects-acne-redness-and-bags-under-the-eyes/
│   │   └── skin-problems-34-on-the-iga-scale/
│   └── reports/
│       ├── step5_face_detection_summary.md
│       ├── step6_deduplication_summary.md
│       ├── step8_split_summary.md
│       └── step9_resnet50_baseline_report.md
├── scripts/
│   ├── scin_download_filtered.py       # SCIN external test set downloader
│   ├── generate_galleries.py           # HTML review galleries generator
│   ├── run_face_detection.py           # MediaPipe face detector script
│   ├── extract_patches.py              # Pre-extracts 224x224 skin patches
│   ├── deduplicate.py                  # pHash deduplication & subject clustering
│   ├── split_dataset.py                # 70/15/15 grouped stratified splitter
│   └── train_baseline.py               # ResNet-50 baseline training script
└── src/
    ├── detection.py                    # MediaPipe face detection & patch extraction
    ├── dataset.py                      # PyTorch Dataset and DataLoader definitions
    ├── model.py                        # ResNet-50 model wrapper & patch aggregation
    ├── inference.py                    # End-to-end inference on unseen raw images
    ├── manifest.py                     # Manifest validation & schema utilities
    └── sources/                        # Dataset-specific ingestion loaders
        ├── scin.py
        ├── iga34.py
        ├── yolo_acne.py
        ├── skin_assessment.py
        ├── skin_defects.py
        ├── roboflow_rosacea.py
        └── roboflow_face_skin.py
```
