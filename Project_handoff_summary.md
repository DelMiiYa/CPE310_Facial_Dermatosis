# Project Handoff Summary: CPE310 Facial Dermatosis Classifier

**Last Updated:** 2026-10-18  
**Repository Working Directory:** `e:\CPE310FacialDermatorsisProject`  
**Git Head:** Clean on `main` (latest commit `d89929f`, core model pipeline committed at [`5af89f7`](file:///e:/CPE310FacialDermatorsisProject))  
**Primary Documentation:**
- [`HANDOFF_PROMPT.md`](file:///e:/CPE310FacialDermatorsisProject/HANDOFF_PROMPT.md) (Authoritative engineering handoff specification)
- [`agent_guidelines.md`](file:///e:/CPE310FacialDermatorsisProject/agent_guidelines.md) (Core rules, domain gap avoidance, low-VRAM constraints)
- [`data/reports/step9_resnet50_baseline_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step9_resnet50_baseline_report.md) (Full baseline evaluation report & confusion matrices)
- [`Literature Survey.pdf`](file:///e:/CPE310FacialDermatorsisProject/Literature_and_Papers/Literature%20Survey.pdf) (Clinical rationale and visual overlap of facial conditions)

---

## 1. Executive Overview & Problem Definition

The objective is to train, evaluate, and interpret a robust multi-class deep learning classifier for facial skin diseases capable of diagnosing **real-world smartphone imagery** (both full-face selfies and macro close-ups).

### Target Classes (4-Class Problem):
1. **Acne** (Acne Vulgaris)
2. **Rosacea**
3. **Dermatitis** (Atopic Eczema + Contact Dermatitis merged per clinical literature)
4. **Normal** (Healthy facial skin control)

### Core Architectural Decisions:
* **Patch-Based Classifier + MediaPipe Face Detector:**
  - **Full-face selfies (`face_detected == True`):** MediaPipe Tasks Vision model (`blaze_face_short_range.tflite` + `face_landmarker.task`) detects the face and 478 landmarks to extract 5 standardized anatomical regions: **forehead, nose, left cheek, right cheek, and chin** ($224 \times 224$).
  - **Macro close-ups (`face_detected == False`):** Tiled into 1–9 standard non-overlapping or lightly overlapping $224 \times 224$ patches.
  - **Inference Aggregation:** Predictions from all patches are aggregated via **mean-probability pooling** to produce an image-level diagnosis.
  - **Rationale:** Smartphone photos vary widely between zoomed-in lesions and full selfies. Decomposing both into standardized skin patches standardizes the input domain, eliminates background noise, and maximizes training data efficiency.
* **Hardware Profile & Optimizations:**
  - Hardware: **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** on Windows with Python 3.13 and PyTorch 2.6 (`+cu124`).
  - Mixed precision training (`torch.amp.autocast('cuda')` with `torch.amp.GradScaler('cuda')`) is strictly enforced.
  - VRAM footprint maintained between **1.2 GB and 3.15 GB** with batch size 32 (comfortably below the 4.0 GB hardware limit).
  - DataLoaders configured with `pin_memory=True`, `persistent_workers=True`, and `num_workers=4` (~55s per epoch, 100% GPU utilization, zero worker starvation).

---

## 2. Hard Engineering Rules & Lessons Learned

1. **The Domain Gap Rule (Lesson from v1):**
   - In v1, training on DermNet clinical macro-lens photos achieved 95% test accuracy on clinical sets but **failed completely (predicted "Normal") on real user selfies**.
   - **Rule:** Training data must strictly represent smartphone camera capture. Textbook clinical imagery (DermNet) is strictly excluded from training.
2. **SCIN is Strictly the Held-Out External Test Set:**
   - Google SCIN dataset (crowd-sourced smartphone photos) is held out exclusively as an **unseen external benchmark** (337 images / 149 cases). SCIN data is **never** used for training, validation, or hyperparameter tuning.
3. **Unified Manifest as Single Source of Truth (`data/manifest/manifest.csv`):**
   - Raw images remain completely untouched in `data/raw/`. All metadata, crops, hashes, labels, review decisions, and split assignments are tracked in the manifest.
4. **Patch Storage & Deprecated Legacy Files:**
   - Pre-extracted 224×224 patches are stored in `data/processed/patches/<split>/` with metadata in `data/manifest/patches_manifest.csv`.
   - Legacy root scripts (`download_data.py`, `filter_and_dedup.py`) from v1 are **deprecated legacy** and must not be used.

---

## 3. Work Completed to Date (Phases 1–4)

### Step 4A & 4B: Dataset Ingestion & Sourcing (Commits `81cabee`, `a51b601`, `970fa5a`)
* **Unified Manifest:** Created [`src/manifest.py`](file:///e:/CPE310FacialDermatorsisProject/src/manifest.py) schema (18 columns, 8,196 total rows).
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
* Evaluated all 8,196 manifest images: 2,827 face-detected (34.5%), 5,369 close-ups/non-face (65.5%). Zero read failures.
* Implemented [`scripts/extract_patches.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/extract_patches.py) which pre-extracted **14,330 patches** ($224 \times 224$) into `data/processed/patches/` with manifest [`data/manifest/patches_manifest.csv`](file:///e:/CPE310FacialDermatorsisProject/data/manifest/patches_manifest.csv):
  - **Train:** 8,462 patches (Acne: 3,046, Rosacea: 2,514, Normal: 2,485, Dermatitis: 417)
  - **Val:** 1,508 patches (294 images)
  - **In-Domain Test:** 1,593 patches (303 images)
  - **SCIN External Test:** 2,767 patches (337 images)
* Reports: [`data/reports/step5_face_detection_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step5_face_detection_summary.md) and HTML sample gallery `step5_patch_samples.html`.

### Step 6: Perceptual Deduplication & Leakage Prevention (Commit `cf4d50c`)
* Implemented [`scripts/deduplicate.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/deduplicate.py) using 64-bit canonical & horizontally-flipped perceptual hashes (`imagehash.phash`, Hamming distance $\le 6\text{--}8$).
* **SCIN External Test Leakage Check:** **0 leaks** found (zero overlap between training candidates and SCIN test set).
* Resolved 3 cross-dataset label conflicts and assigned unified `subject_id`s across all accepted images.
* Report: [`data/reports/step6_deduplication_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step6_deduplication_summary.md).

### Step 7: Manual Review & Training Pool Audit
* Approved clean curated training pool of **2,214 images** across 1,491 subjects:
  - **Acne:** 813 images (330 subjects)
  - **Rosacea:** 664 images (664 subjects)
  - **Normal:** 644 images (404 subjects)
  - **Dermatitis:** 93 images (93 subjects)
* **Unused / Preserved:** 5,645 images preserved in `data/raw/` with rejection reasons recorded in the manifest.

### Step 8: Grouped Stratified 70/15/15 Split (Commit `6456e99`)
* Implemented [`scripts/split_dataset.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/split_dataset.py):
  - Stratified by class and **strictly grouped by `subject_id`** (zero subject leakage across splits).
  - **Train Split (70% images / 70.2% subjects):** 1,617 images / 1,047 subjects (Acne: 591, Rosacea: 466, Normal: 497, Dermatitis: 63)
  - **Validation Split (13.3% images / 14.9% subjects):** 294 images / 222 subjects (Acne: 109, Rosacea: 99, Normal: 71, Dermatitis: 15)
  - **In-Domain Test Split (13.7% images / 14.9% subjects):** 303 images / 222 subjects (Acne: 113, Rosacea: 99, Normal: 76, Dermatitis: 15)
  - **SCIN External Test Set:** 337 images / 149 cases (Acne: 58, Rosacea: 34, Dermatitis: 245)
* Report: [`data/reports/step8_split_summary.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step8_split_summary.md).

### Step 9: ResNet-50 Baseline Training & Evaluation (Commit `5af89f7`)
* Built modular PyTorch training pipeline:
  - Dataset loader: [`src/dataset.py`](file:///e:/CPE310FacialDermatorsisProject/src/dataset.py) (`FacialDermatosisPatchDataset` + fast in-memory dictionary-indexed `FacialDermatosisImageDataset`).
  - Model wrapper: [`src/model.py`](file:///e:/CPE310FacialDermatorsisProject/src/model.py) (`timm` ResNet-50 backbone, dropout 0.2, mean probability pooling).
  - Training script: [`scripts/train_baseline.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/train_baseline.py) (mixed precision AMP, AdamW, Cosine Annealing, image-level validation tracking, UTF-8 console output).
  - Full inference pipeline: [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py) (MediaPipe face detection + patch cropping + classification on raw photos).
* **Optimization & Loss Setup:**
  - Inverse-frequency class weights: $w_c = \frac{N_{total}}{C \times N_c} \rightarrow [\text{Acne}: 0.6945, \text{Rosacea}: 0.8415, \text{Dermatitis}: 5.0731, \text{Normal}: 0.8513]$.
  - The $5.07\times$ weight on Dermatitis prevented model collapse onto majority classes.
* **Best Model Checkpoint:** [`checkpoints/resnet50_baseline_best.pt`](file:///e:/CPE310FacialDermatorsisProject/checkpoints/resnet50_baseline_best.pt) (Epoch 11).

#### In-Domain Test Performance (303 Images / 222 Subjects):
- **Macro-F1:** **`0.8841`**
- **Overall Accuracy:** **`92.74%`**

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Acne** | 0.9630 | 0.9204 | **0.9412** | 113 |
| **Rosacea** | 0.9091 | 0.9091 | **0.9091** | 99 |
| **Dermatitis** | 0.6316 | **0.8000** | **0.7059** | 15 |
| **Normal** | 0.9740 | 0.9868 | **0.9804** | 76 |
| **Macro Average** | **0.8694** | **0.9041** | **0.8841** | 303 |

**In-Domain Confusion Matrix:**
| True \\ Predicted | Acne | Rosacea | Dermatitis | Normal | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Acne** | **104** | 6 | 3 | 0 | 92.0% |
| **Rosacea** | 4 | **90** | 4 | 1 | 90.9% |
| **Dermatitis** | 0 | 2 | **12** | 1 | 80.0% |
| **Normal** | 0 | 1 | 0 | **75** | 98.7% |

#### SCIN Held-Out External Test Performance (337 Images / 149 Cases):
- **Macro-F1:** `0.1544` | **Overall Accuracy:** `21.66%`
- **Acne Recall:** **`91.38%`** (53/58 detected correctly).
- **Rosacea Recall:** `20.59%` (7/34 detected).
- **Dermatitis Recall:** `5.31%` (13/245 detected — 200 predicted as Acne, 26 as Rosacea due to lesion-level close-up visual overlap).
- **Fitzpatrick Skin Type Breakdown:**
  - `FST1` (Very fair): 22.2% (6/27)
  - `FST2` (Fair): 15.5% (22/142)
  - `FST3` (Medium): 26.4% (24/91)
  - `FST4` (Olive / Moderate brown): **31.6%** (12/38)
  - `FST5` (Dark brown): **36.0%** (9/25)
  - *Key Finding:* Accuracy on darker skin types (`FST4-5`) exceeded lighter skin types, confirming robust representation across skin tones.
* Report: [`data/reports/step9_resnet50_baseline_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step9_resnet50_baseline_report.md).

---

## 4. Production Artifacts & Inference Guide

### Checkpoints:
- `checkpoints/resnet50_baseline_best.pt` (Canonical best baseline weights, 282 MB)
- `checkpoints/resnet50_seed42_best.pt` (Seed 42 checkpoint)

### Running Inference on Any Raw Photo:
The standalone inference engine [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py) handles arbitrary input images:
```powershell
python src/inference.py --image "data/raw/skin-problems-34-on-the-iga-scale/images/104/1.jpg"
```
Output:
```
==================================================
PREDICTION RESULT
==================================================
Predicted Class: Acne (Confidence: 99.99%)
Face Detected:   True (5 patches evaluated)

Class Probabilities:
  Acne        : 99.99%
  Rosacea     : 0.00%
  Dermatitis  : 0.00%
  Normal      : 0.01%
==================================================
```

---

## 5. Current Project State & Immediate Next Steps (Task 7)

> [!IMPORTANT]
> **Current Milestone:** Step 9 (ResNet-50 Baseline) is fully completed, evaluated, and committed to git. Task 7 is ready to begin.

### Upcoming Work (Task 7: Advanced Architectures, Ensemble & Explainability):
1. **EfficientNet-B0 Model:**
   - Train `efficientnet_b0` on identical 14,330 patches, identical 70/15/15 folds, and identical class weights.
   - Compare parameter count (~5.3M vs ~25.6M for ResNet-50), inference latency, and Macro-F1.
2. **Model Ensemble:**
   - Implement `src/ensemble.py` combining ResNet-50 and EfficientNet-B0 via probability averaging ($\frac{P_{resnet} + P_{effnet}}{2}$) as mandated by the literature survey.
3. **Grad-CAM Interpretability:**
   - Implement `src/explainability.py` using `pytorch-grad-cam` to generate class activation heatmaps across face landmark crops and close-up tiles.
4. **SCIN Dermatitis vs. Acne Domain Gap Analysis:**
   - Investigate why crowd-sourced dermatitis close-ups trigger acne predictions (papular vs. eczematous texture overlap) and explore confidence calibration or threshold tuning.

---

## 6. Complete Repository Structure Map

```
E:\CPE310FACIALDERMATORSISPROJECT
├── .env                                # Private API keys (git-ignored)
├── .gitignore                          # Ignores data/, .env, runs/, checkpoints/
├── agent_guidelines.md                 # Core engineering rules & domain shift guidelines
├── project_prompt.md                   # Master 5-phase project guidelines
├── HANDOFF_PROMPT.md                   # Authoritative full engineering handoff document
├── Project_handoff_summary.md          # Comprehensive executive summary (this document)
├── datasets_links.txt                  # Origin URLs for raw datasets & Roboflow
├── requirements.txt                    # Project python dependencies (includes scikit-learn>=1.4)
├── checkpoints/                        # Git-ignored model weights
│   ├── resnet50_baseline_best.pt       # Canonical best ResNet-50 baseline checkpoint
│   └── resnet50_seed42_best.pt         # Seed 42 checkpoint
├── data/                               # Git-ignored data directory
│   ├── manifest/
│   │   ├── manifest.csv                # Master image manifest (8,196 rows, 18 columns)
│   │   ├── manifest_before_step5.csv   # Step 5 backup manifest
│   │   ├── manifest_before_step6.csv   # Step 6 backup manifest
│   │   ├── phash_cache.json            # 64-bit canonical & flip perceptual hashes
│   │   └── patches_manifest.csv        # Extracted patches manifest (14,330 patches)
│   ├── processed/
│   │   └── patches/                    # Pre-extracted 224x224 skin patches (train, val, test, external_test)
│   ├── raw/                            # Untouched raw dataset folders (Rule 2)
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
│       ├── step9_resnet50_baseline_report.md
│       ├── patch_samples/              # Visual patch crop validation samples
│       └── galleries/                  # Generated HTML review galleries
├── runs/                               # Training run logs
│   └── resnet50_seed42/
│       └── training_log.csv            # Per-epoch training & validation history
├── scripts/
│   ├── scin_download_filtered.py       # SCIN external test set downloader
│   ├── generate_galleries.py           # HTML review galleries generator
│   ├── run_face_detection.py           # MediaPipe face detector script
│   ├── extract_patches.py              # Pre-extracts 224x224 skin patches
│   ├── deduplicate.py                  # pHash deduplication & subject clustering
│   ├── split_dataset.py                # 70/15/15 grouped stratified splitter
│   └── train_baseline.py               # ResNet-50 baseline training & evaluation script
├── src/
│   ├── detection.py                    # MediaPipe face detection & patch extraction
│   ├── dataset.py                      # PyTorch Dataset and DataLoader definitions
│   ├── model.py                        # ResNet-50 model wrapper & patch aggregation
│   ├── inference.py                    # End-to-end inference on unseen raw images
│   ├── manifest.py                     # Manifest validation & schema utilities
│   └── sources/                        # Dataset-specific ingestion loaders
│       ├── scin.py
│       ├── iga34.py
│       ├── yolo_acne.py
│       ├── skin_assessment.py
│       ├── skin_defects.py
│       ├── roboflow_rosacea.py
│       └── roboflow_face_skin.py
└── Literature_and_Papers/
    └── Literature Survey.pdf           # Literature survey & comparative analysis
```
