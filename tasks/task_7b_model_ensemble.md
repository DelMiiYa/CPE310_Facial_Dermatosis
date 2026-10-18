# Goldfish Task Handoff: Task 7B — ResNet-50 + EfficientNet-B0 Model Ensemble

*Paste this file into a fresh Goldfish worker agent, or run a subagent pointed at this file. It is 100% self-contained.*

---

## 1. Goldfish Role & Strict Constraints

You are a stateless **Goldfish execution agent** working under the **Elephant & Goldfish** methodology on the CPE310 Facial Dermatosis Classification project (`e:\CPE310FacialDermatorsisProject`).
Your sole responsibility is **Task 7B: Implementing, evaluating, and benchmarking the ResNet-50 + EfficientNet-B0 Model Ensemble** against both single-model baselines across the Validation set, In-Domain Test set, and Held-Out SCIN External Test set.

### Hard Engineering Rules (DO NOT VIOLATE):
1. **Scope Boundary:** Execute **ONLY Task 7B**. Do not implement Grad-CAM (Task 7C) or SCIN post-hoc calibration (Task 7D). Do **not** retrain ResNet-50 or EfficientNet-B0 — both models are already trained and saved in `checkpoints/`.
2. **Untouched Raw Data, Splits & Existing Checkpoints:**
   - Never modify `data/raw/`, `data/manifest/`, or `data/processed/patches/`.
   - Never overwrite `checkpoints/resnet50_baseline_best.pt`, `checkpoints/resnet50_seed42_best.pt`, `checkpoints/efficientnet_b0_baseline_best.pt`, `checkpoints/efficientnet_b0_seed42_best.pt`, `data/reports/step9_resnet50_baseline_report.md`, or `data/reports/step10_efficientnet_b0_report.md`.
3. **Zero Test Leakage in Weight Tuning:**
   - Any ensemble weight tuning ($w_{\text{resnet}}$ vs. $w_{\text{effnet}}$) MUST be performed strictly on the **Validation Split (`val`, 294 images)**. Never tune ensemble weights on `test` or `external_test`.
4. **Hardware & Low-VRAM Guardrails (RTX 3050 4GB VRAM, Windows, Python 3.13):**
   - Explicitly use `torch.amp.autocast("cuda")` during forward passes.
   - Never use `cd` in PowerShell commands; run all commands with `e:\CPE310FacialDermatorsisProject` as working directory.
   - Never print or log `.env` or `ROBOFLOW_API_KEY`.

---

## 2. Context & Single-Model Baseline Reference (Entering Task 7B)

Both single-backbone classifiers have been trained and evaluated on the exact same 70/15/15 split (`seed=42`):

| Metric | ResNet-50 (`resnet50_baseline_best.pt`) | EfficientNet-B0 (`efficientnet_b0_baseline_best.pt`) | Notes / Complementarity |
|---|---|---|---|
| **Parameters / Checkpoint** | 23.52M / 269.6 MB (Epoch 11) | 4.01M / 46.4 MB (Epoch 9) | Combined footprint: 27.53M / 316.0 MB |
| **Best Val Image Macro-F1** | `0.8897` | **`0.9227`** | EfficientNet-B0 +3.30% on `val` (294 imgs) |
| **In-Domain Test Macro-F1** | **`0.8841`** | `0.8644` | ResNet-50 higher on `Dermatitis` recall |
| **In-Domain Test Accuracy** | **`92.74%`** (281/303) | **`92.74%`** (281/303) | Identical accuracy, different errors! |
| **In-Domain Acne F1** | `0.9412` (P: 0.9630, R: 0.9204) | **`0.9464`** (P: 0.9550, R: 0.9381) | EfficientNet-B0 stronger on Acne |
| **In-Domain Rosacea F1** | `0.9091` (P: 0.9091, R: 0.9091) | **`0.9100`** (P: 0.9010, R: 0.9192) | EfficientNet-B0 slightly higher recall |
| **In-Domain Dermatitis F1** | **`0.7059`** (P: 0.6316, R: 0.8000) | `0.6207` (P: 0.6429, R: 0.6000) | ResNet-50 higher recall (12/15 vs 9/15); EffNet higher precision |
| **In-Domain Normal F1** | **`0.9804`** (75/76) | **`0.9804`** (75/76) | Both 98.68% recall |
| **SCIN External Test Macro-F1**| **`0.1544`** (Acc: `21.66%`) | `0.1477` (Acc: `19.58%`) | EffNet Rosacea F1 is **`0.2500`** vs ResNet `0.2000` (+5.0%) |

### Disagreement & Oracle Potential (from Task 7A Audit):
On the 303 In-Domain Test images:
- **Both Correct:** 275 / 303 (`90.76%`)
- **ResNet-50 Correct ONLY:** 6 images
- **EfficientNet-B0 Correct ONLY:** 6 images
- **Neither Correct:** 16 images
- **Theoretical Oracle Ceiling:** **287 / 303 (`94.72%` accuracy)** — 12 disagreement cases where soft probability fusion can resolve errors!

---

## 3. Step-by-Step Execution Instructions for Goldfish Agent

### Step 1: Implement `src/ensemble.py`
Create [`src/ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/src/ensemble.py) providing:
1. **`DermatosisEnsemble(nn.Module)`:**
   - Accepts checkpoints for ResNet-50 (default `checkpoints/resnet50_baseline_best.pt`) and EfficientNet-B0 (default `checkpoints/efficientnet_b0_baseline_best.pt`), device, and fusion weights `weights: Tuple[float, float] = (0.5, 0.5)`.
   - Loads both `DermatosisClassifier` models via `src/model.py` (`build_model`) and sets them to `.eval()`.
   - **`predict_patches(x: torch.Tensor) -> Dict[str, torch.Tensor]`:**
     - Runs mixed-precision (`torch.amp.autocast`) forward pass on patch batch `x` (`K x 3 x 224 x 224`) through both models.
     - Computes `resnet_probs = softmax(resnet_logits, dim=-1)` and `effnet_probs = softmax(effnet_logits, dim=-1)`.
     - Computes weighted patch probabilities:
       $$P_{\text{ens\_patch}} = w_{\text{resnet}} P_{\text{resnet\_patch}} + w_{\text{effnet}} P_{\text{effnet\_patch}}$$
     - Returns a dict with `"ensemble_probs"`, `"resnet_probs"`, and `"effnet_probs"`.
   - **`predict_image_from_patches(x: torch.Tensor, aggregation: str = "mean") -> Dict[str, Any]`:**
     - Aggregates `resnet_probs` -> `resnet_img_probs` ($4$-dim), `effnet_probs` -> `effnet_img_probs` ($4$-dim), and fuses:
       $$P_{\text{ens\_img}} = w_{\text{resnet}} P_{\text{resnet\_img}} + w_{\text{effnet}} P_{\text{effnet\_img}}$$
       *(Note: For linear mean pooling across patches, fusing at the patch level and then mean-pooling is mathematically identical to mean-pooling each model's patch probabilities and then fusing at the image level! Supporting cached image-level probabilities makes validation weight grid-search instantaneous).*
2. **`find_optimal_ensemble_weights(resnet_img_probs: np.ndarray, effnet_img_probs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]`:**
   - Performs a 1D grid search over $w_{\text{resnet}} \in \{0.00, 0.05, 0.10, \dots, 1.00\}$ (with $w_{\text{effnet}} = 1 - w_{\text{resnet}}$) strictly on the **Validation set (294 images)** to maximize Validation Image-Level Macro-F1 (tie-breaking on Validation Accuracy, then closeness to `0.50`).

### Step 2: Integrate `--ensemble` into `src/inference.py`
Update [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py) (preserving 100% backward compatibility for single-model `--checkpoint` calls):
- Add `--ensemble` flag (and optional `--weights` float pair, defaulting to `0.5 0.5`) to `src/inference.py`.
- When `--ensemble` is passed (or via `FacialDermatosisPredictor(..., ensemble=True)`), load `DermatosisEnsemble`, run MediaPipe face detection / patch extraction, and include in the returned dictionary and CLI printout:
  - Ensemble `predicted_class`, `confidence`, and `class_probabilities`
  - Individual backbone predictions (`resnet50_prediction`, `efficientnet_b0_prediction`, and whether the two models agreed).
- Test `python src/inference.py --image "data/raw/skin-problems-34-on-the-iga-scale/images/104/1.jpg" --ensemble` to confirm clean execution.

### Step 3: Create & Run `scripts/evaluate_ensemble.py`
Create [`scripts/evaluate_ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/evaluate_ensemble.py) that:
1. Evaluates `resnet50`, `efficientnet_b0`, **Equal-Weight Ensemble (`0.50 / 0.50`)**, and **Val-Tuned Ensemble ($w^*_{\text{resnet}} / w^*_{\text{effnet}}$)** across:
   - `val` (294 images)
   - `test` (In-Domain Test, 303 images)
   - `external_test` (SCIN Held-Out Benchmark, 337 images)
2. Benchmarks **RTX 3050 GPU Inference Latency** (warm up 20 passes, time 100 passes with `torch.cuda.synchronize()`) for 1-patch (`1x3x224x224`), 5-patch selfie (`5x3x224x224`), and full 303-image test set throughput (img/s and ms/img) for the Ensemble compared to single models.
3. Analyzes the **12 Disagreement Cases** on the 303 In-Domain Test images (where ResNet-50 and EfficientNet-B0 predicted different classes) and reports exactly how many of the 12 disagreements the Ensemble resolved correctly!
4. Saves detailed per-image prediction CSVs to `runs/ensemble_seed42/`:
   - `runs/ensemble_seed42/val_predictions.csv`
   - `runs/ensemble_seed42/test_predictions.csv`
   - `runs/ensemble_seed42/scin_predictions.csv`
   *(Include columns: `image_id`, `subject_id`, `source`, `true_label`, `resnet_pred`, `effnet_pred`, `ens_equal_pred`, `ens_tuned_pred`, `models_agree`, plus the 4 class probabilities for `resnet`, `effnet`, and `ens_equal` — these CSVs will directly power Task 7C Grad-CAM case selection and Task 7D SCIN calibration analysis!).*
5. Generates the comprehensive markdown report [`data/reports/step11_ensemble_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step11_ensemble_report.md) containing:
   - **Section 1:** Executive Summary & 4-Way Comparison Table (`ResNet-50` vs. `EfficientNet-B0` vs. `Ensemble (0.5/0.5 Equal)` vs. `Ensemble (Val-Tuned)` on Params, Disk Size, 5-Patch Latency, Val Macro-F1, In-Domain Test Macro-F1 & Accuracy, SCIN External Test Macro-F1 & Accuracy).
   - **Section 2:** Validation Weight Tuning Curve ($w_{\text{resnet}} \in [0.0, 1.0]$ table showing Val Macro-F1 and Val Accuracy).
   - **Section 3:** In-Domain Test Per-Class Breakdown (`Acne`, `Rosacea`, `Dermatitis`, `Normal` Precision, Recall, F1) & Confusion Matrices for the Ensemble vs. Single Models.
   - **Section 4:** Disagreement Resolution Analysis (breakdown of the 12 test images where ResNet-50 and EfficientNet-B0 disagreed, showing how soft probability averaging resolves high-confidence vs. low-confidence conflicts).
   - **Section 5:** SCIN Held-Out External Test Per-Class Breakdown, Confusion Matrix, and Fitzpatrick Skin Type (`FST1`–`FST5`) Accuracy Comparison across all models.

---

## 4. Acceptance Criteria & Handoff Back to Elephant

Before finishing, verify all items below:
- [ ] [`src/ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/src/ensemble.py) and [`scripts/evaluate_ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/evaluate_ensemble.py) are implemented and run cleanly.
- [ ] [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py) supports `--ensemble` while remaining backward-compatible with single-checkpoint inference.
- [ ] `runs/ensemble_seed42/{val,test,scin}_predictions.csv` exist on disk.
- [ ] [`data/reports/step11_ensemble_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step11_ensemble_report.md) exists and contains all 5 required sections.
- [ ] Existing checkpoints (`resnet50_*`, `efficientnet_b0_*`) and raw data (`data/raw/`) are completely untouched.
- [ ] Return a concise summary of the final Ensemble metrics to the Elephant coordinator.
