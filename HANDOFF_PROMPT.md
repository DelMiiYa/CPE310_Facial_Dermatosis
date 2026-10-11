# Handoff Prompt: Facial Dermatosis Classification (CPE310)

*Paste this entire file into the next AI assistant. It is completely self-contained.*

---

## 1. Your Role & Working Rules

You are an expert ML engineer continuing an **existing** end-to-end PyTorch computer-vision project. Previous assistants completed project setup, SCIN external test set ingestion, local dataset inventory, visual gallery generation, and manual review. **Do not redo completed steps, and do not reverse approved decisions.**

**Working rules (strict):**
1. Work step by step. **Propose each step and wait for the user's approval before running it.** Never write the whole codebase at once.
2. **Never delete or modify anything in `data/raw/`.** Never use DermNet or clinical-only imagery for training.
3. Commit to the local git repo after each completed step (`data/` and `.env` are git-ignored).
4. Stop at every **⏸ STOP** point and let the user review.
5. Read `agent_guidelines.md` and `project_prompt.md` in the project root. They contain hard rules:
   - Domain gap avoidance (selfies/smartphones only for training)
   - Deduplication via pHash Hamming distance $\le 6\text{--}8$ before splitting
   - 70/15/15 stratified split grouped by `subject_id`
   - Evaluation on Macro-F1
   - ResNet-50 baseline first
   - 3 random seeds & Grad-CAM verification.

---

## 2. Project Context & Hardware

- **Goal:** Multi-class classifier for facial skin conditions from **smartphone photos** (accepting full-face selfies and close-ups).
- **Target Classes:** `Acne`, `Rosacea`, `Dermatitis` (atopic + contact merged), `Normal`.
- **Project root:** `E:\CPE310FacialDermatorsisProject` (Windows, PowerShell, Python 3.13).
- **Hardware:** NVIDIA RTX 3050 Laptop GPU, **4 GB VRAM**. Verified CUDA working (`torch 2.6.0+cu124`).
- **Low-VRAM requirements:** Mixed precision (`torch.amp.autocast`), batch size 16–32, DataLoaders with `pin_memory=True`, `persistent_workers=True`, `num_workers=4`.
- **Architecture decision:** Pretrained face detector (MediaPipe) + single patch classifier:
  - Face detected (selfie): Crop skin regions (forehead, nose, left cheek, right cheek, chin) and classify patches.
  - No face detected (close-up): Tile image into patches.
  - Aggregate patch scores into overall diagnosis + Grad-CAM heatmap.

---

## 3. Approved Decisions & Lessons Learned

1. **SCIN is strictly the external test set:** Never used for training, validation, or tuning. 339 facial images downloaded to `data/raw/scin/dataset/images/`.
2. **The Domain Gap Lesson (CRITICAL):** In v1, training on DermNet clinical photos yielded 95% test accuracy on DermNet but 0% on real selfies. Training data must strictly look like smartphone photos.
3. **Discovery regarding `skin-assessment-dataset`:** Manual review and file inspection confirmed that `skin_assessment` is mostly repackaged DermNet (`07rosacea...`, `03eczema...`) plus non-face body parts. It **cannot** be trusted as our primary training pool for Rosacea or Dermatitis without domain contamination.
4. **Manifest single source of truth:** `data/manifest/manifest.csv` tracks all images without copying raw files.

---

## 4. Current State of the Repository

- **Git history:**
  - `690cee8`: Initial handoff prompt and Step 4 spec
  - `81cabee`: Step 4 completion (Manifest schema, SCIN test set download, local source loaders, HTML gallery generation)
  - `b98438c`: Ignore `.env` and private keys in `.gitignore`
  - `548bd15`: User added Roboflow rosacea dataset link in `datasets_links.txt`
- **Environment & Keys:**
  - Roboflow API key is stored securely in `.env` (`ROBOFLOW_API_KEY=...`) and git-ignored.
  - `roboflow` Python SDK 1.7.1 installed.
  - `mediapipe`, `opencv`, `grad-cam`, `timm`, `torch` installed with CUDA support.
- **Manifest Status (`data/manifest/manifest.csv`):**
  - Total registered images: 6,809
  - External test (SCIN): 337 images (Acne: 58, Dermatitis: 245, Rosacea: 34)
  - Train pool: 6,472 images (`iga34`: 687 acne, `yolo_acne`: 927 acne, `skin_defects`: 30 acne, `skin_assessment`: 4,828 mixed)

---

## 5. User Review Feedback (Step 4C Results)

The user reviewed the generated HTML galleries (`data/reports/galleries/`) and gave the following verdict:
1. **Labels:** Broadly accurate.
2. **Domain & Non-face issue:** `scin - Dermatitis`, `skin_assessment - Dermatitis`, and `skin_assessment - Normal` contain non-face body photos (arms, legs, hands, torso). MediaPipe face filtering (Step 5) is required to filter out non-face imagery.
3. **Data Gap Resolution:**
   - The user found a dedicated Roboflow dataset specifically for Rosacea and added it to line 9 of `datasets_links.txt`:
     `https://universe.roboflow.com/skin-face-ixq7s/rosacea-n6xhv-jnjno/`
   - In addition, project `facial-skin-dataset/face-skin-diseases` was forked into the user's Roboflow workspace `d3l-m11` as `d3l-m11/face-skin-diseases-bbveb` (498 raw images: 100 rosacea, 99 eksim, 100 acne).

---

## 6. Immediate Next Tasks for the Next Assistant

### Task 1: Ingest Roboflow Rosacea Dataset (Solve the Rosacea Gap)
- Check `datasets_links.txt` line 9 (`skin-face-ixq7s/rosacea-n6xhv-jnjno`).
- Write a download script using `roboflow` SDK or API (reading `ROBOFLOW_API_KEY` from `.env`) to download the images into `data/raw/roboflow-rosacea/`. Ensure unaugmented/high-resolution images are fetched.
- Write a source loader in `src/sources/roboflow_rosacea.py` to register them in `data/manifest/manifest.csv` with `label=Rosacea`, `role=train_pool`.
- Inspect a small gallery of them to confirm they look like smartphone photos/close-ups before committing.

### Task 2: Step 5 - Face Detection & Region Cropping Pipeline (MediaPipe)
- Build `src/detection.py` and `scripts/run_face_detection.py`:
  - Run MediaPipe face detector over all images in the manifest.
  - Update `face_detected` boolean in `manifest.csv`.
  - For images where `face_detected == False`, determine if they are valid close-up skin crops or non-face body parts (arms/legs/hands) that should have `review_status = rejected`.
  - Implement the patch extraction logic:
    - Selfie (face detected): crop patches for forehead, nose, left cheek, right cheek, and chin.
    - Close-up (facial skin patch): tile into $224 \times 224$ patches.

### Task 3: Step 6 - Perceptual Hashing & Deduplication
- Implement `scripts/deduplicate.py` using `imagehash.phash`.
- Deduplicate across ALL sources with Hamming distance $\le 6\text{--}8$.
- Critical rule: Deduplicate **before** splitting to prevent data leakage. Flag any near-duplicates with conflicting class labels.

### Task 4: Step 7 - Manual Review (⏸ STOP Point)
- Present sample cropped patches and deduplication results to the user for review.

### Task 5: Step 8 & 9 - Stratified Split & ResNet-50 Baseline
- Split 70/15/15 grouped by `subject_id` and stratified by class.
- Train ResNet-50 baseline with class-weighted cross-entropy and mixed precision, evaluated on Macro-F1 across 3 seeds.

**Start by briefly acknowledging this handoff, confirming the next action (Task 1: Fetching the user's selected Rosacea dataset from Roboflow), and asking for the user's approval to proceed.**
