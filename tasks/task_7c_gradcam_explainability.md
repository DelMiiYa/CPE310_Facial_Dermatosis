# Goldfish Task Handoff: Task 7C — Grad-CAM Visual Interpretability & Saliency Analysis

*Paste this file into a fresh Goldfish worker agent, or run a subagent pointed at this file. It is 100% self-contained.*

---

## 1. Goldfish Role & Strict Constraints

You are a stateless **Goldfish execution agent** working under the **Elephant & Goldfish** methodology on the CPE310 Facial Dermatosis Classification project (`e:\CPE310FacialDermatorsisProject`).
Your sole responsibility is **Task 7C: Implementing Grad-CAM visual interpretability (`src/explainability.py` & `scripts/generate_gradcam.py`) across ResNet-50, EfficientNet-B0, and their 0.5/0.5 Ensemble, generating an interactive HTML visual gallery and interpretability report**.

### Hard Engineering Rules (DO NOT VIOLATE):
1. **Scope Boundary:** Execute **ONLY Task 7C**. Do not implement SCIN post-hoc calibration or temperature scaling (Task 7D). Do **not** retrain any models.
2. **Untouched Raw Data, Splits & Existing Checkpoints:**
   - Never modify `data/raw/`, `data/manifest/`, or `data/processed/patches/`.
   - Never overwrite existing checkpoints in `checkpoints/`, prediction CSVs in `runs/ensemble_seed42/`, or reports `step9_*.md`, `step10_*.md`, `step11_*.md`.
3. **Hardware & Low-VRAM Guardrails (RTX 3050 4GB VRAM, Windows, Python 3.13):**
   - Never use `cd` in PowerShell commands; run all commands with `e:\CPE310FacialDermatorsisProject` as working directory.
   - Never print or log `.env` or `ROBOFLOW_API_KEY`.
   - **Important Grad-CAM Technical Note:** `pytorch-grad-cam` requires autograd gradients (`with torch.enable_grad():`) and standard `float32` tensors (do **not** wrap `GradCAM` calls inside `torch.no_grad()` or `torch.amp.autocast`). Since Grad-CAM evaluates 1–9 patches of a single image at a time, `float32` uses `< 500 MB` VRAM on the RTX 3050.

---

## 2. Context & Codebase Reference (Entering Task 7C)

The repository contains trained single models, the dual-backbone ensemble, and full per-image predictions from Task 7B:

- **Classes (`src/dataset.py`):** `["Acne", "Rosacea", "Dermatitis", "Normal"]` (`NUM_CLASSES = 4`).
- **Trained Checkpoints:**
  - `checkpoints/resnet50_baseline_best.pt` (`timm` `resnet50` wrapped in `DermatosisClassifier`, where `classifier.model` is the `timm` ResNet-50 model).
    - **Target Layer for `pytorch-grad-cam`:** `[classifier.model.layer4[-1]]` (final bottleneck block producing $7 \times 7$ spatial feature maps).
  - `checkpoints/efficientnet_b0_baseline_best.pt` (`timm` `efficientnet_b0` wrapped in `DermatosisClassifier`, where `classifier.model` is the `timm` EfficientNet-B0 model).
    - **Target Layer for `pytorch-grad-cam`:** `[classifier.model.conv_head]` (or `classifier.model.bn2`, final spatial conv block producing $7 \times 7$ feature maps before global average pooling).
- **Manifests & Pre-Extracted Patches:**
  - `data/manifest/manifest.csv`: Master image manifest (`image_id`, `rel_path` relative to `data/raw/`, `label`, `split`, `face_detected`, etc.). Note: check how `src/dataset.py` or `scripts/extract_patches.py` resolves raw image paths if needed, or use `data/manifest/patches_manifest.csv` which stores every extracted $224 \times 224$ patch (`image_id`, `patch_idx`, `patch_type`, `region_name`, `patch_rel_path` relative to `data/processed/patches/` or `patch_path`).
- **Task 7B Ensemble Prediction Logs (`runs/ensemble_seed42/`):**
  - `runs/ensemble_seed42/test_predictions.csv` (303 In-Domain Test images with `resnet_pred`, `effnet_pred`, `ens_equal_pred`, `models_agree`, and all class probabilities).
  - `runs/ensemble_seed42/scin_predictions.csv` (337 SCIN External Test images with `fst` and all class probabilities).

---

## 3. Step-by-Step Execution Instructions for Goldfish Agent

### Step 1: Implement `src/explainability.py`
Create [`src/explainability.py`](file:///e:/CPE310FacialDermatorsisProject/src/explainability.py) using `pytorch_grad_cam` (`from pytorch_grad_cam import GradCAM`, `from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget`, `from pytorch_grad_cam.utils.image import show_cam_on_image`):
1. **Target Layer Helper (`get_target_layers(classifier: DermatosisClassifier)`):**
   - For `resnet50`: returns `[classifier.model.layer4[-1]]`.
   - For `efficientnet_b0`: returns `[classifier.model.conv_head]`.
2. **`DermatosisExplainer` Class:**
   - Loads both `resnet50` (`checkpoints/resnet50_baseline_best.pt`) and `efficientnet_b0` (`checkpoints/efficientnet_b0_baseline_best.pt`) onto `device` in `.eval()` mode (with parameters `requires_grad_(True)` so Grad-CAM can compute gradients with respect to activations).
   - Instantiates `GradCAM(model=self.resnet, target_layers=get_target_layers(self.resnet))` and `GradCAM(model=self.effnet, target_layers=get_target_layers(self.effnet))`.
   - **`explain_patch_batch(patch_tensors: torch.Tensor, patch_rgb_floats: List[np.ndarray], target_class_idx: Optional[int] = None) -> Dict[str, Any]`:**
     - Given $K$ normalized patch tensors (`K x 3 x 224 x 224`) and their corresponding unnormalized RGB float arrays (`[0, 1]`, `224 x 224 x 3`):
     - Computes ResNet-50 probabilities, EfficientNet-B0 probabilities, and Equal-Ensemble ($0.5/0.5$) probabilities.
     - If `target_class_idx` is `None`, targets the **Ensemble predicted class index** (or allows specifying any target class index so we can inspect what triggered a predicted class).
     - Computes `resnet_cams` (`K x 224 x 224` in `[0, 1]`), `effnet_cams` (`K x 224 x 224` in `[0, 1]`), and **Fused Ensemble CAM**:
       $$\text{CAM}_{\text{ens}} = 0.5 \cdot \text{CAM}_{\text{resnet}} + 0.5 \cdot \text{CAM}_{\text{effnet}}$$
       (re-normalized to `[0, 1]` per patch where max > 0).
     - Produces RGB heatmap overlays (`224 x 224 x 3`, uint8) via `show_cam_on_image(rgb_float, cam, use_rgb=True)` for ResNet-50, EfficientNet-B0, and Ensemble.
     - Computes a quantitative **Patch Attention Attribution Score** for each patch $k \in \{1, \dots, K\}$:
       - Mean CAM activation intensity ($\bar{A}_k = \text{mean}(\text{CAM}_{\text{ens}, k})$) and top-quartile activation intensity, plus the patch's predicted probability for the target class, identifying the **Dominant Diagnostic Patch** (e.g., `left_cheek` vs `nose` vs `forehead`).
   - **`explain_raw_image(image_path: Path, output_png_path: Optional[Path] = None)`:**
     - Runs MediaPipe face detection & patch extraction (via `src/detection.py`) on a raw image, computes Grad-CAM on all extracted patches, projects the patch CAMs back onto the original image's bounding boxes `(x1, y1, x2, y2)` to create a full-image heatmap overlay alongside the individual patch heatmaps, and optionally saves the visualization to `output_png_path`.
3. **Optional CLI in `src/explainability.py`:**
   - Allow running `python src/explainability.py --image <path> --output <out.png>` on any raw photo.

### Step 2: Create & Run `scripts/generate_gradcam.py` on Curated Diagnostic Cohort
Create [`scripts/generate_gradcam.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/generate_gradcam.py) that selects **16 curated, representative images** across 4 diagnostic cohorts using `runs/ensemble_seed42/test_predictions.csv`, `runs/ensemble_seed42/scin_predictions.csv`, and `data/manifest/patches_manifest.csv`:

1. **Cohort A — In-Domain True Positive Full-Face Selfies (4 images: 1 `Acne`, 1 `Rosacea`, 1 `Dermatitis`, 1 `Normal` where `face_detected == True` / 5 anatomical patches):**
   - Demonstrates anatomical region localization (`forehead`, `nose`, `left_cheek`, `right_cheek`, `chin`) and confirms the models focus on skin pathology rather than eyes, hair, or background.
2. **Cohort B — In-Domain True Positive Close-Up Macro Tiles (4 images across `Acne`, `Rosacea`, `Dermatitis` where `face_detected == False`):**
   - Demonstrates lesion-level attention on inflammatory papules/pustules, malar erythema, and eczematous plaques.
3. **Cohort C — Model Disagreement & Conflict Resolution Cases (4 images from the 15 test disagreements in `runs/ensemble_seed42/test_predictions.csv`):**
   - Include:
     - `roboflow_rosacea_328f0448f636` (True: `Rosacea` | ResNet: `Rosacea 71.2%`, EffNet: `Dermatitis 51.0%` -> **Ensemble resolved `Rosacea 60.1%`**)
     - `roboflow_face_skin_e224eda66ed6` (True: `Acne` | ResNet: `Dermatitis 39.7%`, EffNet: `Acne 72.2%` -> **Ensemble resolved `Acne 51.4%`**)
     - `roboflow_face_skin_d42a51631dc4` (True: `Rosacea` | ResNet: `Dermatitis 60.4%`, EffNet: `Rosacea 89.9%` -> **Ensemble resolved `Rosacea 62.9%`**)
     - `roboflow_face_skin_fba72dde5494` (True: `Dermatitis` | ResNet: `Dermatitis 54.0%`, EffNet: `Acne 86.2%` -> **Ensemble fooled into `Acne 61.2%`**)
   - For each case, visualize side-by-side: Original Patches, ResNet-50 Grad-CAM (for its predicted class), EfficientNet-B0 Grad-CAM (for its predicted class), and Ensemble Grad-CAM!
4. **Cohort D — SCIN Held-Out External Test Benchmark Cases (4 images from `runs/ensemble_seed42/scin_predictions.csv`):**
   - Include:
     - 1 True Positive SCIN `Acne` case on darker skin tone (`FST4` or `FST5`, e.g., `scin_bd88f2d86eba` or `scin_aea2a7b30663`).
     - 1 Ensemble-Synergy SCIN `Rosacea` case: `scin_d7fef8551dbd` (True: `Rosacea`, `FST4` — where ResNet predicted `Acne 34.7%`, EffNet predicted `Normal 46.2%`, and **Ensemble correctly predicted `Rosacea 35.7%`**!).
     - 2 Misclassified SCIN `Dermatitis -> Acne` cases (e.g., `scin_bb48c000df7e` where both models predict Acne >94%, and `scin_a2060741433b` where ResNet correctly predicted `Dermatitis 36.8%` but EffNet's `Acne 71.4%` pulled the Ensemble to `Acne`). This directly visualizes what focal papular/erythematous textures cause the SCIN Dermatitis-to-Acne domain shift!

### Step 3: Save Visualizations, Quantitative Region Statistics & HTML Gallery
In `scripts/generate_gradcam.py`:
1. Save high-resolution multi-panel PNG figures for all 16 cases to `data/reports/gradcam_samples/<cohort>_<image_id>.png` (each figure clearly titled with `image_id`, True Label, ResNet/EffNet/Ensemble predictions & confidences, and columns showing: Original Patch, ResNet-50 CAM, EfficientNet-B0 CAM, and Ensemble Fused CAM across the image's patches).
2. Also compute **Quantitative Anatomical Region Importance Statistics** across the full **In-Domain Test Set (`face_detected == True` subset)** for each disease class (`Acne`, `Rosacea`, `Dermatitis`, `Normal`):
   - Which of the 5 anatomical patches (`forehead`, `nose`, `left_cheek`, `right_cheek`, `chin`) has the highest mean target-class probability and highest Grad-CAM activation per condition? (e.g., verifying clinically whether `Rosacea` focuses most strongly on `left_cheek`, `right_cheek`, and `nose` — the classic malar/centrofacial distribution!).
3. Generate an interactive, self-contained HTML gallery at [`data/reports/galleries/step12_gradcam_gallery.html`](file:///e:/CPE310FacialDermatorsisProject/data/reports/galleries/step12_gradcam_gallery.html) embedding/linking the 16 multi-panel PNGs organized by Cohort A, B, C, and D with diagnostic captions.
4. Generate the comprehensive markdown report at [`data/reports/step12_gradcam_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step12_gradcam_report.md) containing:
   - **Section 1:** Executive Summary & Methodology (Target layers for ResNet-50 and EfficientNet-B0, patch-level vs. full-face saliency reconstruction, Fused Ensemble CAM formulation).
   - **Section 2:** Quantitative Anatomical Region Attribution Analysis (Table showing average class probability & Grad-CAM activation across `forehead`, `nose`, `left_cheek`, `right_cheek`, `chin` per disease class on face-detected test selfies, confirming clinical alignment such as centrofacial/malar dominance in Rosacea and T-zone/cheek dominance in Acne).
   - **Section 3:** Qualitative Cohort Breakdown (Cohorts A, B, C, D findings across all 16 cases, linking to `data/reports/gradcam_samples/*.png` and `step12_gradcam_gallery.html`).
   - **Section 4:** Key Mechanistic Insights for Task 7D (What the Grad-CAM heatmaps on Cohort C & D reveal about *why* EfficientNet-B0 is overconfident on papular textures in Dermatitis images, and how temperature scaling / calibration in Task 7D can address it).

---

## 4. Acceptance Criteria & Handoff Back to Elephant

Before finishing, verify all items below:
- [ ] [`src/explainability.py`](file:///e:/CPE310FacialDermatorsisProject/src/explainability.py) and [`scripts/generate_gradcam.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/generate_gradcam.py) are implemented and run cleanly without errors.
- [ ] All 16 multi-panel Grad-CAM PNGs exist in `data/reports/gradcam_samples/`.
- [ ] Interactive HTML gallery exists at [`data/reports/galleries/step12_gradcam_gallery.html`](file:///e:/CPE310FacialDermatorsisProject/data/reports/galleries/step12_gradcam_gallery.html).
- [ ] Comprehensive report exists at [`data/reports/step12_gradcam_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step12_gradcam_report.md) with all 4 sections and quantitative anatomical region attribution table.
- [ ] Existing checkpoints, manifests, and raw data (`data/raw/`) are completely untouched.
- [ ] Save a handoff summary to `tasks/task_7c_handoff_report.md` and return a concise summary of findings to the Elephant coordinator.
