# Goldfish Task Handoff: Task 7A — EfficientNet-B0 Training & Benchmark Comparison

*Paste this file into a fresh Goldfish worker agent, or run a subagent pointed at this file. It is 100% self-contained.*

---

## 1. Goldfish Role & Strict Constraints

You are a stateless **Goldfish execution agent** working under the **Elephant & Goldfish** methodology on the CPE310 Facial Dermatosis Classification project (`e:\CPE310FacialDermatorsisProject`).
Your sole responsibility is **Task 7A: Training and evaluating the EfficientNet-B0 patch-based classifier and comparing it head-to-head against the completed ResNet-50 baseline**.

### Hard Engineering Rules (DO NOT VIOLATE):
1. **Scope Boundary:** Execute **ONLY Task 7A**. Do not implement ensembles (Task 7B), Grad-CAM (Task 7C), or SCIN calibration (Task 7D).
2. **Untouched Raw Data & Splits:** Never modify `data/raw/`, `data/manifest/manifest.csv`, `data/manifest/patches_manifest.csv`, or `data/processed/patches/`. Use the existing pre-extracted 14,330 patches and 70/15/15 splits as-is.
3. **Preserve ResNet-50 Baseline:** Do not overwrite `checkpoints/resnet50_baseline_best.pt`, `checkpoints/resnet50_seed42_best.pt`, `runs/resnet50_seed42/`, or `data/reports/step9_resnet50_baseline_report.md`.
4. **Hardware & Low-VRAM Guardrails (RTX 3050 4GB VRAM, Windows, Python 3.13):**
   - Explicitly use `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.
   - Use `batch_size=32`, `num_workers=4`, `pin_memory=True`, `persistent_workers=True`.
   - Never use `cd` in PowerShell commands; run all commands with `e:\CPE310FacialDermatorsisProject` as working directory.
   - Never print or log `.env` or `ROBOFLOW_API_KEY`.

---

## 2. Context & Existing Codebase Reference

The repository already contains the full patch dataset and modular PyTorch training pipeline built in Step 9:

- **Classes (`src/dataset.py`):** `["Acne", "Rosacea", "Dermatitis", "Normal"]` (`NUM_CLASSES = 4`).
- **Pre-extracted Patches (`data/manifest/patches_manifest.csv`):**
  - `train`: 8,462 patches (1,617 images / 1,047 subjects)
  - `val`: 1,508 patches (294 images / 222 subjects)
  - `test` (In-Domain Test): 1,593 patches (303 images / 222 subjects)
  - `external_test` (SCIN Held-Out Benchmark): 2,767 patches (337 images / 149 cases)
- **Model Wrapper (`src/model.py`):**
  - `DermatosisClassifier(backbone_name=..., num_classes=4, pretrained=True, drop_rate=0.2)` already uses `timm.create_model(...)`! Passing `backbone_name="efficientnet_b0"` works out of the box via `build_model(model_name="efficientnet_b0", ...)`.
- **Baseline Script (`scripts/train_baseline.py`):**
  - Currently trains `resnet50` with `AdamW(lr=1e-4, weight_decay=1e-2)`, `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`, inverse-frequency class weights `compute_class_weights(split="train")` ($w_c = [0.6945, 0.8415, 5.0731, 0.8513]$), early stopping (`patience=6`) on **Validation Image-Level Macro-F1** (mean-probability pooling across patches).
- **ResNet-50 Reference Metrics (`data/reports/step9_resnet50_baseline_report.md`):**
  - **Best Epoch:** 11 (Seed 42)
  - **In-Domain Test (303 imgs):** Macro-F1 = **`0.8841`**, Accuracy = **`92.74%`**
    - Per-Class F1: Acne `0.9412` (P: `0.9630`, R: `0.9204`), Rosacea `0.9091` (P: `0.9091`, R: `0.9091`), Dermatitis `0.7059` (P: `0.6316`, R: `0.8000`), Normal `0.9804` (P: `0.9740`, R: `0.9868`).
  - **SCIN External Test (337 imgs):** Macro-F1 = **`0.1544`**, Accuracy = **`21.66%`**
    - Per-Class Recall: Acne `91.38%` (F1 `0.3183`), Rosacea `20.59%` (F1 `0.2000`), Dermatitis `5.31%` (F1 `0.0992`).
    - Fitzpatrick Accuracy: FST1 `22.2%`, FST2 `15.5%`, FST3 `26.4%`, FST4 `31.6%`, FST5 `36.0%`.

---

## 3. Step-by-Step Execution Instructions for Goldfish Agent

### Step 1: Parameterize Training & Benchmarking Script
Either extend [`scripts/train_baseline.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/train_baseline.py) with a `--backbone` CLI argument (defaulting to `resnet50` for backward compatibility) **or** create a dedicated `scripts/train_efficientnet.py` that imports shared helpers from `scripts/train_baseline.py` / `src/`.
Ensure that when training `efficientnet_b0`:
1. Hyperparameters, seed (`42`), epochs (`15`), batch size (`32`), learning rate (`1e-4`), weight decay (`1e-2`), dropout (`0.2`), and class weights are **identical** to ResNet-50 for a fair scientific comparison.
2. Checkpoints are saved to:
   - `checkpoints/efficientnet_b0_seed42_best.pt`
   - `checkpoints/efficientnet_b0_baseline_best.pt`
   *(Ensure `model_name="efficientnet_b0"` is also stored inside the checkpoint state dict).*
3. Per-epoch logs are saved to `runs/efficientnet_b0_seed42/training_log.csv`.
4. Also support `--model-name` / auto-detection in [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py) so `src/inference.py` can run with either ResNet-50 or EfficientNet-B0 checkpoints seamlessly.

### Step 2: Train EfficientNet-B0 (Seed 42)
Run the training script on GPU (`cuda`) for Seed 42 (15 epochs, early stopping patience 6) and verify:
- CUDA is active and VRAM usage stays well below 4 GB.
- Training converges cleanly without NaN losses.

### Step 3: Measure Efficiency Metrics (Parameters, Size, Latency)
Measure and compare both `resnet50` (`checkpoints/resnet50_baseline_best.pt`) and `efficientnet_b0` (`checkpoints/efficientnet_b0_baseline_best.pt`) on the RTX 3050 GPU:
1. **Total Parameters & Trainable Parameters** (millions).
2. **Checkpoint File Size** on disk (MB).
3. **Inference Latency:**
   - Warm up GPU with 20 forward passes, then time 100 forward passes (with `torch.cuda.synchronize()`) for:
     - Single patch batch (`1 x 3 x 224 x 224`) and 5-patch selfie batch (`5 x 3 x 224 x 224`) in ms.
     - End-to-end throughput on the 303 In-Domain Test images (images/sec and ms/image).

### Step 4: Generate Comparative Report (`data/reports/step10_efficientnet_b0_report.md`)
Generate a comprehensive markdown report at `data/reports/step10_efficientnet_b0_report.md` containing:
1. **Executive Summary & Architecture Comparison Table:**
   - Side-by-side table comparing `ResNet-50` vs. `EfficientNet-B0` on: Params (M), Checkpoint Size (MB), 5-Patch GPU Latency (ms), Best Val Macro-F1, **In-Domain Test Macro-F1**, **In-Domain Test Accuracy**, **SCIN External Test Macro-F1**, and **SCIN External Test Accuracy**.
2. **EfficientNet-B0 In-Domain Test Per-Class Breakdown & Confusion Matrix** (with delta comparison vs. ResNet-50 per class: Acne, Rosacea, Dermatitis, Normal).
3. **EfficientNet-B0 SCIN External Test Breakdown, Confusion Matrix & Fitzpatrick Skin Type (FST1–FST5) Table.**
4. **Key Observations for Task 7B (Ensemble):** Note where EfficientNet-B0 and ResNet-50 have complementary strengths (e.g., whether EfficientNet-B0 improves Dermatitis or Rosacea precision/recall).

---

## 4. Acceptance Criteria & Handoff Back to Elephant

Before finishing, verify all items below:
- [ ] `checkpoints/efficientnet_b0_baseline_best.pt` and `checkpoints/efficientnet_b0_seed42_best.pt` exist on disk.
- [ ] `runs/efficientnet_b0_seed42/training_log.csv` exists and has complete epoch history.
- [ ] `data/reports/step10_efficientnet_b0_report.md` exists and contains all required tables.
- [ ] `src/inference.py` works with `--checkpoint checkpoints/efficientnet_b0_baseline_best.pt`.
- [ ] `git status` shows zero modifications to `data/raw/` or existing ResNet-50 checkpoints/reports.
- [ ] Return a concise summary of the final metrics to the Elephant coordinator.
