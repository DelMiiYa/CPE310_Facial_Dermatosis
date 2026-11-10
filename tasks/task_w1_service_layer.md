# Goldfish Task Spec: W1 — Service Layer & Validation Abstention Threshold

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier (Acne, Rosacea, Dermatitis, Normal) using MediaPipe face/patch extraction and a temperature-scaled ResNet-50 + EfficientNet-B0 ensemble (`T_resnet=0.60, T_effnet=1.20, w_resnet=0.40`).
All ML training and calibration are complete. We are now wrapping the validated pipeline into a web application.
Task W1 builds the backend inference service (`webapp/service.py`) and the validation-split abstention threshold derivation script (`scripts/pick_abstention_threshold.py`).
This service loads the models ONCE onto the GPU (or CPU fallback) without duplicating weights between inference and Grad-CAM, and returns the frozen `/api/analyze` JSON contract dictionary.

## 2. Goal
Create `scripts/pick_abstention_threshold.py` to compute the 85%-coverage validation abstention threshold from `runs/ensemble_seed42/val_predictions.csv`, and create `webapp/service.py` exposing `WebAppDermatosisService` and `analyze(image_bytes)` conforming to the frozen API contract.

## 3. Files to Read First
Read these exact files before writing code:
- `tasks/WEBAPP_MASTER_PLAN.md`
- `src/dataset.py`
- `src/detection.py`
- `src/ensemble.py`
- `src/calibration.py`
- `src/explainability.py`
- `runs/ensemble_seed42/val_predictions.csv` (first 10 lines to verify column names)

### Copied Function Signatures from `src/` (Verified)
```python
# src/dataset.py
CLASS_NAMES: List[str] = ["Acne", "Rosacea", "Dermatitis", "Normal"]
CLASS_TO_IDX: Dict[str, int] = {"Acne": 0, "Rosacea": 1, "Dermatitis": 2, "Normal": 3}
IDX_TO_CLASS: Dict[int, str] = {0: "Acne", 1: "Rosacea", 2: "Dermatitis", 3: "Normal"}
get_eval_transforms(img_size: int = 224) -> transforms.Compose  # input: uint8 RGB ndarray (224,224,3) -> Tensor(3,224,224)

# src/detection.py
class FaceDetector:
    def __init__(self, model_path: Optional[Path] = None, min_detection_confidence: float = 0.5): ...
    def detect(self, image_or_path: Any, max_dim: int = 1280) -> Dict[str, Any]: ...
    def close(self) -> None: ...

class FaceLandmarker:
    def __init__(self, model_path: Optional[Path] = None): ...
    def detect_landmarks(self, image_bgr: np.ndarray) -> Optional[List[Tuple[float, float, float]]]: ...
    def close(self) -> None: ...

def extract_face_patches(
    image_bgr: np.ndarray,
    detection_info: Optional[Dict[str, Any]] = None,
    landmarker: Optional[FaceLandmarker] = None,
    target_size: int = 224,
) -> Dict[str, Any]:
    # Returns {"patches": Dict[str, np.ndarray], "boxes": Dict[str, Tuple[int, int, int, int]]}
    # Keys in order: ["forehead", "nose", "left_cheek", "right_cheek", "chin"]

def extract_tile_patches(
    image_bgr: np.ndarray,
    target_size: int = 224,
    max_tiles: int = 9,
) -> Dict[str, Any]:
    # Returns {"patches": List[np.ndarray], "boxes": List[Tuple[int, int, int, int]], "image_tiled": np.ndarray}

# src/ensemble.py
DEFAULT_RESNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet50_baseline_best.pt"
DEFAULT_EFFNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_baseline_best.pt"

class DermatosisEnsemble(nn.Module):
    def __init__(
        self,
        resnet_checkpoint: Union[str, Path] = DEFAULT_RESNET_CHECKPOINT,
        effnet_checkpoint: Union[str, Path] = DEFAULT_EFFNET_CHECKPOINT,
        weights: Tuple[float, float] = (0.5, 0.5),
        device: Optional[Union[str, torch.device]] = None,
    ): ...
    def predict_image_from_patches(self, x: torch.Tensor, aggregation: str = "mean") -> Dict[str, Any]:
        # Returns dict with keys:
        # "ensemble_probs": Tensor(4,), "resnet_probs": Tensor(4,), "effnet_probs": Tensor(4,),
        # "pred_idx": int, "predicted_class": str, "resnet_pred_idx": int, "resnet_predicted_class": str,
        # "effnet_pred_idx": int, "effnet_predicted_class": str, "models_agree": bool,
        # "ensemble_patch_probs": Tensor(K, 4), "resnet_patch_probs": Tensor(K, 4), "effnet_patch_probs": Tensor(K, 4)

# src/calibration.py
def predict_calibrated_ensemble(
    res_probs: np.ndarray,
    eff_probs: np.ndarray,
    w_resnet: float,
    t_resnet: float,
    t_effnet: float,
) -> np.ndarray: ...

def apply_bayes_prior_adjustment(
    probs: np.ndarray,          # shape (N, 4) — MUST be 2D!
    source_priors: np.ndarray,  # shape (4,)
    target_priors: np.ndarray,  # shape (4,)
    eps: float = 1e-12,
) -> np.ndarray: ...

def evaluate_selective_prediction(
    probs: np.ndarray,
    targets: np.ndarray,
    coverages: Optional[List[float]] = None,
    uncertainty_metric: str = "confidence",
) -> List[Dict[str, Any]]: ...

# src/explainability.py
def get_target_layers(classifier: Union[DermatosisClassifier, nn.Module]) -> List[nn.Module]: ...
# From pytorch_grad_cam:
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
```

### Copied Frozen Contract from `tasks/WEBAPP_MASTER_PLAN.md`
`analyze(image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]` must return:
```json
{
  "prediction": "rosacea",
  "prediction_label": "Rosacea",
  "confidence": 0.9124,
  "abstained": false,
  "abstention_threshold": 0.8548,
  "abstention_message": null,
  "caution_note": null,
  "triage_mode": false,
  "probabilities": {"acne": 0.0412, "rosacea": 0.9124, "dermatitis": 0.0281, "normal": 0.0183},
  "top2": ["rosacea", "acne"],
  "input_mode": "face",
  "face_detected": true,
  "num_patches": 5,
  "dominant_region": "left_cheek",
  "region_probabilities": {
    "forehead": {"acne": 0.081, "rosacea": 0.842, "dermatitis": 0.045, "normal": 0.032},
    "nose": {"acne": 0.032, "rosacea": 0.931, "dermatitis": 0.021, "normal": 0.016},
    "left_cheek": {"acne": 0.029, "rosacea": 0.945, "dermatitis": 0.018, "normal": 0.008},
    "right_cheek": {"acne": 0.034, "rosacea": 0.928, "dermatitis": 0.024, "normal": 0.014},
    "chin": {"acne": 0.03, "rosacea": 0.916, "dermatitis": 0.0325, "normal": 0.0215}
  },
  "model_breakdown": {
    "resnet50": {
      "prediction": "rosacea",
      "confidence": 0.931,
      "probabilities": {"acne": 0.031, "rosacea": 0.931, "dermatitis": 0.022, "normal": 0.016}
    },
    "efficientnet_b0": {
      "prediction": "rosacea",
      "confidence": 0.9,
      "probabilities": {"acne": 0.048, "rosacea": 0.9, "dermatitis": 0.0322, "normal": 0.0198}
    },
    "models_agree": true,
    "calibration_params": {"t_resnet": 0.6, "t_effnet": 1.2, "w_resnet": 0.4, "w_effnet": 0.6}
  },
  "original_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,...",
  "latency_ms": 124.5,
  "info": {},
  "disclaimer": "Educational research demonstration only — NOT a medical diagnosis. Do not use for clinical decision-making. Please consult a qualified dermatologist."
}
```

## 4. Files to Create or Edit
Create ONLY these files (everything else is strictly off limits):
1. `scripts/pick_abstention_threshold.py`
2. `webapp/__init__.py`
3. `webapp/service.py`

## 5. Step-by-Step Instructions

### Step 1: Write and run `scripts/pick_abstention_threshold.py`
1. Load `runs/ensemble_seed42/val_predictions.csv` (294 validation images) using `pandas`.
2. Extract `val_res = df[[f"resnet_prob_{c}" for c in CLASS_NAMES]].values.astype(np.float64)` and `val_eff = df[[f"effnet_prob_{c}" for c in CLASS_NAMES]].values.astype(np.float64)`, and `val_targets = np.array([CLASS_TO_IDX[l] for l in df["true_label"]], dtype=np.int64)`.
3. Compute calibrated validation probabilities using `predict_calibrated_ensemble(val_res, val_eff, w_resnet=0.40, t_resnet=0.60, t_effnet=1.20)` from `src/calibration.py`.
4. Call `evaluate_selective_prediction(val_cal_probs, val_targets, coverages=[1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70], uncertainty_metric="confidence")`.
5. Print a clean markdown table of Coverage (%), Retained Count (`k/total`), Abstention Count, Threshold (`tau` to 4 decimal places), Validation Accuracy (%), and Validation Macro-F1.
6. Also print the exact recommended constant line for **85% validation coverage** (the midpoint of the 80–90% target range):
   `ABSTENTION_THRESHOLD: float = <tau_at_0.85_rounded_to_4_decimals>`
7. Run `python scripts/pick_abstention_threshold.py` and copy that exact threshold value into `webapp/service.py`.

### Step 2: Create `webapp/__init__.py`
Create an empty `webapp/__init__.py` (with a one-line docstring).

### Step 3: Create `webapp/service.py`
Implement the following constants, exception classes, helper functions, and service class in `webapp/service.py`:

1. **Constants:**
   - `SYSTEM_NAME: str = "DermaScan Face AI"`
   - `CALIBRATION_T_RESNET: float = 0.60`
   - `CALIBRATION_T_EFFNET: float = 1.20`
   - `CALIBRATION_W_RESNET: float = 0.40`
   - `CALIBRATION_W_EFFNET: float = 0.60`
   - `ABSTENTION_THRESHOLD: float = <exact 85% val coverage threshold from Step 1>` (add a comment documenting the exact val accuracy, val Macro-F1, and coverage from `scripts/pick_abstention_threshold.py`).
   - `TRAINING_SOURCE_PRIORS = np.array([591 / 1617, 466 / 1617, 63 / 1617, 497 / 1617], dtype=np.float64)`
   - `TRIAGE_TARGET_PRIORS = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float64)`
   - `MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024`
   - `ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}`
   - `ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}`
   - `FACE_REGIONS = ["forehead", "nose", "left_cheek", "right_cheek", "chin"]`
   - `DISCLAIMER_TEXT = "Educational research demonstration only — NOT a medical diagnosis. Do not use for clinical decision-making. Please consult a qualified dermatologist."`

2. **Custom Exceptions for Friendly Error Handling:**
   ```python
   class WebAppInputError(ValueError):
       def __init__(self, detail: str, detail_th: str, error_code: str, status_code: int = 400):
           super().__init__(detail)
           self.detail = detail
           self.detail_th = detail_th
           self.error_code = error_code
           self.status_code = status_code
   ```
   Provide helper validation functions or raise `WebAppInputError` with exact `error_code` and `status_code`:
   - Empty bytes (`len(image_bytes) == 0`): `error_code="empty_file"`, `status_code=400`
   - Oversized bytes (`len(image_bytes) > MAX_UPLOAD_BYTES`): `error_code="file_too_large"`, `status_code=413`
   - Corrupted/unreadable image (`cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)` is `None`): `error_code="unreadable_image"`, `status_code=400`
   - No face AND no usable skin region (`has_usable_skin_region(img_bgr)` is `False` when `face_detected == False`): `error_code="no_skin_detected"`, `status_code=422`.

3. **Usable Skin Region Check (`has_usable_skin_region(img_bgr: np.ndarray) -> Tuple[bool, float]`):**
   When `face_detected == False` (the close-up path), `extract_tile_patches` would otherwise tile any non-skin image (blank image, screenshot, landscape, etc.). Implement `has_usable_skin_region(img_bgr)`:
   - If `img_bgr is None` or `min(img_bgr.shape[:2]) < 64`: return `(False, 0.0)`.
   - Compute grayscale standard deviation `gray_std = float(np.std(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))`. If `gray_std < 8.0` (solid/blank image): return `(False, 0.0)`.
   - Convert `img_bgr` to YCrCb and HSV.
   - A pixel is in the broad human-skin / erythema gamut (covering Fitzpatrick I–VI and inflamed erythematous lesions) if:
     - YCrCb: `(Y >= 20) & (Y <= 245) & (Cr >= 125) & (Cr <= 188) & (Cb >= 70) & (Cb <= 145)`
     - **AND** HSV saturation is non-grayscale: `(S >= 12) & (S <= 230) & (V >= 25)`
     - **AND** HSV hue is in warm/skin/red/brown range: `((H <= 32) | (H >= 155))` (note OpenCV `H` is in `[0, 180]`).
   - Compute `skin_frac = float(np.mean(skin_mask))`.
   - Return `(skin_frac >= 0.15, skin_frac)`.

4. **Single-Copy Model Loading & Thread Safety (`WebAppDermatosisService`):**
   - In `__init__(self, device: Optional[str] = None)`:
     - Use a `threading.Lock()` (`self._lock`) so GPU inference and Grad-CAM hooks process one request at a time safely within the 4 GB VRAM budget.
     - Instantiate `self.ensemble = DermatosisEnsemble(weights=(CALIBRATION_W_RESNET, CALIBRATION_W_EFFNET), device=device)` ONCE.
     - Set `self.ensemble.resnet.requires_grad_(True)` and `self.ensemble.effnet.requires_grad_(True)` so Grad-CAM can compute gradients on the **exact same model instances** without loading a second copy of ResNet-50 or EfficientNet-B0!
     - Instantiate `self.cam_resnet = GradCAM(model=self.ensemble.resnet, target_layers=get_target_layers(self.ensemble.resnet))` and `self.cam_effnet = GradCAM(model=self.ensemble.effnet, target_layers=get_target_layers(self.ensemble.effnet))`.
     - Instantiate `self.detector = FaceDetector()` and `self.landmarker = FaceLandmarker()`.
     - Instantiate `self.eval_transforms = get_eval_transforms(img_size=224)`.
     - Load `webapp/knowledge_base.json` if it exists on disk (re-loading or caching it; if `webapp/knowledge_base.json` does not exist yet during parallel W1 testing, gracefully return `{}` for `"info"` and reload when `knowledge_base.json` becomes available).

5. **Exact Inference + Calibration + Grad-CAM Pipeline in `analyze(self, image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]`:**
   - Start timer `t0 = time.perf_counter()`.
   - Validate `image_bytes` (non-empty, `<= MAX_UPLOAD_BYTES`, decodes via `cv2.imdecode`).
   - Under `with self._lock:`:
     1. Run `det_info = self.detector.detect(img_bgr)`. `is_face = bool(det_info.get("face_detected", False))`.
     2. If `is_face`:
        - Call `face_res = extract_face_patches(img_bgr, detection_info=det_info, landmarker=self.landmarker, target_size=224)`.
        - Order regions as `FACE_REGIONS = ["forehead", "nose", "left_cheek", "right_cheek", "chin"]`.
        - `canvas_bgr = img_bgr`, `input_mode = "face"`.
     3. If not `is_face`:
        - Check `ok_skin, skin_frac = has_usable_skin_region(img_bgr)`. If not `ok_skin`, raise `WebAppInputError` with `error_code="no_skin_detected"`, `status_code=422`.
        - Call `tile_res = extract_tile_patches(img_bgr, target_size=224, max_tiles=9)`.
        - `canvas_bgr = tile_res["image_tiled"]`, `input_mode = "closeup"`.
     4. Build `stacked_tensors` `(K, 3, 224, 224)` using `self.eval_transforms(cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB))`.
     5. Call `ens_raw = self.ensemble.predict_image_from_patches(stacked_tensors, aggregation="mean")` (which runs under `@torch.no_grad()` and `torch.amp.autocast`).
        - Extract image-level `r_img_probs = ens_raw["resnet_probs"].cpu().numpy()` `(4,)` and `e_img_probs = ens_raw["effnet_probs"].cpu().numpy()` `(4,)`.
        - Extract patch-level `r_patch_probs = ens_raw["resnet_patch_probs"].cpu().numpy()` `(K, 4)` and `e_patch_probs = ens_raw["effnet_patch_probs"].cpu().numpy()` `(K, 4)`.
     6. Compute calibrated image-level probabilities:
        ```python
        cal_img_probs = predict_calibrated_ensemble(
            r_img_probs[np.newaxis, :],
            e_img_probs[np.newaxis, :],
            w_resnet=CALIBRATION_W_RESNET,
            t_resnet=CALIBRATION_T_RESNET,
            t_effnet=CALIBRATION_T_EFFNET,
        )[0]  # shape (4,)
        ```
        And calibrated patch-level probabilities:
        ```python
        cal_patch_probs = predict_calibrated_ensemble(
            r_patch_probs,
            e_patch_probs,
            w_resnet=CALIBRATION_W_RESNET,
            t_resnet=CALIBRATION_T_RESNET,
            t_effnet=CALIBRATION_T_EFFNET,
        )  # shape (K, 4)
        ```
     7. If `triage_mode` is `True`:
        ```python
        cal_img_probs = apply_bayes_prior_adjustment(
            cal_img_probs[np.newaxis, :],
            TRAINING_SOURCE_PRIORS,
            TRIAGE_TARGET_PRIORS,
        )[0]
        cal_patch_probs = apply_bayes_prior_adjustment(
            cal_patch_probs,
            TRAINING_SOURCE_PRIORS,
            TRIAGE_TARGET_PRIORS,
        )
        ```
     8. Determine top-1 and top-2 classes from `cal_img_probs`:
        - `sorted_idxs = np.argsort(-cal_img_probs)`
        - `pred_idx = int(sorted_idxs[0])`
        - `prediction_label = CLASS_NAMES[pred_idx]` (e.g. `"Rosacea"`)
        - `prediction = prediction_label.lower()` (e.g. `"rosacea"`)
        - `confidence = round(float(cal_img_probs[pred_idx]), 4)`
        - `top2 = [CLASS_NAMES[int(sorted_idxs[0])].lower(), CLASS_NAMES[int(sorted_idxs[1])].lower()]`
        - `abstained = bool(confidence < ABSTENTION_THRESHOLD)`
        - If `abstained`:
          `abstention_message = f"Uncertain — please consult a dermatologist. Top-2 candidates: {CLASS_NAMES[int(sorted_idxs[0])]} ({cal_img_probs[int(sorted_idxs[0])]*100:.1f}%) and {CLASS_NAMES[int(sorted_idxs[1])]} ({cal_img_probs[int(sorted_idxs[1])]*100:.1f}%)."`
          Else `None`.
        - Set `caution_note` per the Master Plan contract (`"dermatitis"` and `"normal"` notes).
     9. Compute Grad-CAM overlay for `target_class_idx = pred_idx`:
        - Using `x_grad = stacked_tensors.to(self.ensemble.device, dtype=torch.float32)` and `targets = [ClassifierOutputTarget(pred_idx)] * K`:
        - Run `resnet_cams = self.cam_resnet(input_tensor=x_grad, targets=targets)` and `effnet_cams = self.cam_effnet(input_tensor=x_grad, targets=targets)` inside `with torch.enable_grad():`.
        - Fuse each patch CAM `k`: `raw_fuse = 0.5 * resnet_cams[k] + 0.5 * effnet_cams[k]`, normalize each patch CAM to `[0, 1]` (matching `src/explainability.py` lines 206–213), find `dominant_region = patch_names[int(np.argmax([ float(np.mean(ens_cams[k])) for k in range(K) ]))]`.
        - Project `ens_cams` onto `canvas_bgr` `(cH, cW)` using `boxes` (matching `src/explainability.py` lines 363–394), normalize `full_cam_ens` to `[0, 1]`, and produce `full_overlay_rgb = show_cam_on_image(canvas_rgb_float, full_cam_ens, use_rgb=True)`.
        - Explicitly delete `x_grad`, `resnet_cams`, `effnet_cams` and call `self.ensemble.resnet.zero_grad(set_to_none=True)` and `self.ensemble.effnet.zero_grad(set_to_none=True)` so no gradient tensors linger in VRAM.
     10. Encode both `canvas_bgr` (`original_image`) and `cv2.cvtColor(full_overlay_rgb, cv2.COLOR_RGB2BGR)` (`gradcam_image`) to PNG bytes in memory using `cv2.imencode(".png", ...)` and format as `"data:image/png;base64,<b64>"` ( resize `canvas_bgr` and `full_overlay_rgb` proportionally if `max(cH, cW) > 900` before PNG encoding so base64 payloads stay fast and compact).
     11. Assemble and return the exact contract dictionary.
6. Also provide a module-level singleton accessor `get_service() -> WebAppDermatosisService` and top-level convenience function `analyze(image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]` that delegates to `get_service().analyze(image_bytes, triage_mode=triage_mode)`.

## 6. Hard Rules
- **DO NOT** touch `data/raw/`, `checkpoints/`, `.env`, or any existing file in `src/`.
- **DO NOT** retrain or refit calibration parameters on `test` or `scin`. Use `T_resnet=0.60, T_effnet=1.20, w_resnet=0.40`.
- **DO NOT** save uploaded images to disk. Perform all decoding and PNG base64 encoding strictly in memory (`cv2.imdecode` / `cv2.imencode`).
- **Load models ONCE** per `WebAppDermatosisService` instance and share the ResNet-50 and EfficientNet-B0 backbones between `DermatosisEnsemble` and `GradCAM` to stay well under the 4 GB VRAM limit on RTX 3050.
- Use UTF-8 encoding for all files. Do not use `cd` in PowerShell commands; run all commands from `e:\CPE310FacialDermatorsisProject`.

## 7. Acceptance Criteria
Run these two commands from `e:\CPE310FacialDermatorsisProject` and paste their exact output into your handoff report:

1. Validation abstention threshold script:
```powershell
python scripts/pick_abstention_threshold.py
```
Expected: Prints the validation coverage table (100% down to 70%) on the 294 val images and prints `ABSTENTION_THRESHOLD: float = ...` for 85% coverage.

2. Service layer smoke & contract verification (face mode, close-up mode, triage mode, and error handling):
```powershell
python -c "from pathlib import Path; import torch; from webapp.service import get_service, analyze, WebAppInputError; svc = get_service(); img_bytes = Path('data/raw/skin-problems-34-on-the-iga-scale/images/104/1.jpg').read_bytes(); res = analyze(img_bytes); print('Keys:', list(res.keys())); print('Pred:', res['prediction'], 'Conf:', res['confidence'], 'Mode:', res['input_mode'], 'Patches:', res['num_patches'], 'Abstained:', res['abstained'], 'Latency_ms:', res['latency_ms']); print('Probs sum:', round(sum(res['probabilities'].values()), 4)); print('VRAM MB:', round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 'CPU')"
```
Expected: Prints all 19 contract keys, `Pred: acne`, `Mode: face`, `Patches: 5`, `Probs sum: 1.0`, and `VRAM MB:` well below `1500` MB (limit is 4096 MB).

## 8. Stop Conditions
- If any file in `src/` or `runs/ensemble_seed42/val_predictions.csv` is missing or its signature differs from Section 3 of this spec, **STOP** immediately and report the discrepancy in `tasks/task_w1_handoff_report.md`. Do not improvise or modify `src/`.

## 9. Report Instructions
When finished, write `tasks/task_w1_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact commands run and their actual output (pasted verbatim, not summarised)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check
- **Numbers to carry forward:** exact `ABSTENTION_THRESHOLD`, validation accuracy & Macro-F1 at 80%/85%/90% coverage, single-request latency (ms), and peak GPU VRAM (MB).

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session 1)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w1_service_layer.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** The acceptance command printed `Pred: acne`, `Probs sum: 1.0`, and recorded the exact `ABSTENTION_THRESHOLD` from the validation set.
