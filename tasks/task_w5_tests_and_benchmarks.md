# Goldfish Task Spec: W5 — Automated Tests, Consistency Check & Latency/VRAM Benchmark

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier web application (`DermaScan Face AI`).
Tasks W1 (`webapp/service.py`), W2 (`webapp/knowledge_base.json`), W3 (`webapp/app.py`), and W4 (`webapp/static/*`) are complete.
Task W5 creates the comprehensive automated test suite at `tests/test_webapp.py` using `pytest` and FastAPI's `TestClient`.
It verifies all API routes, static files, all 8 demo scenarios, consistency against `runs/ensemble_seed42/test_predictions.csv` across 16 test-split images, and records end-to-end latency and peak GPU VRAM.

## 2. Goal
Create `tests/test_webapp.py` and run `pytest` to verify API smoke tests, all 8 demo scenarios, 16-image consistency against `runs/ensemble_seed42/test_predictions.csv`, and end-to-end latency/VRAM metrics.

## 3. Files to Read First
Read these exact files before writing any code:
- `tasks/WEBAPP_MASTER_PLAN.md`
- `webapp/app.py`
- `webapp/service.py`
- `runs/ensemble_seed42/test_predictions.csv` (first 20 lines)
- `data/manifest/manifest.csv` (first 20 lines)

### Copied Signatures & Constants from `webapp/` and `src/`
```python
# webapp/app.py
from webapp.app import app  # FastAPI instance

# webapp/service.py
from webapp.service import (
    ABSTENTION_THRESHOLD,       # 0.7852
    CALIBRATION_T_RESNET,       # 0.60
    CALIBRATION_T_EFFNET,       # 1.20
    CALIBRATION_W_RESNET,       # 0.40
    CALIBRATION_W_EFFNET,       # 0.60
    MAX_UPLOAD_BYTES,           # 10 * 1024 * 1024
    get_service,
    analyze,
)

# src/calibration.py
from src.calibration import predict_calibrated_ensemble
from src.dataset import CLASS_NAMES, CLASS_TO_IDX  # ["Acne", "Rosacea", "Dermatitis", "Normal"]
```

### Important Recon Note on `test_predictions.csv` vs Raw Image Crops
- `runs/ensemble_seed42/test_predictions.csv` contains 303 rows for `split == "test"`, with columns:
  `image_id, subject_id, source, true_label, resnet_pred, effnet_pred, ens_equal_pred, ens_tuned_pred, models_agree, resnet_prob_Acne, effnet_prob_Acne, ..., resnet_prob_Normal, effnet_prob_Normal`.
- Those CSV probabilities are **uncalibrated** per-backbone probabilities (`resnet_prob_<Class>`, `effnet_prob_<Class>`) computed from pre-extracted patches in `data/processed/patches/test/*.jpg` (which `scripts/extract_patches.py` saved with `IMWRITE_JPEG_QUALITY=95`).
- To get the expected **calibrated** probabilities from `test_predictions.csv`, apply:
  ```python
  cal_csv_probs = predict_calibrated_ensemble(
      res_probs, eff_probs,
      w_resnet=0.40, t_resnet=0.60, t_effnet=1.20
  )
  ```
- When `POST /api/analyze` processes the raw image from `data/raw/...`, it crops patches directly in memory without JPEG-95 recompression, so probabilities match `cal_csv_probs` within a small numerical tolerance (mean absolute difference `< 0.03` across test images, and exact `< 1e-4` when `predict_calibrated_ensemble` is compared directly on patch tensors).

## 4. Files to Create or Edit
Create ONLY this file (everything else is strictly off limits):
1. `tests/test_webapp.py`

## 5. Step-by-Step Instructions

### Step 1: Create `tests/test_webapp.py`
Write a clean, well-structured `pytest` suite in `tests/test_webapp.py` using a module-scoped `TestClient(app)` fixture (`@pytest.fixture(scope="module") def client(): with TestClient(app) as c: yield c`) containing these exact test functions:

1. **`test_health_endpoint(client)`:**
   - `GET /api/health` returns `200`.
   - JSON has `status == "ok"`, `models_loaded is True`, `knowledge_base_loaded is True`, `abstention_threshold == 0.7852`, and `calibration == {"t_resnet": 0.6, "t_effnet": 1.2, "w_resnet": 0.4, "w_effnet": 0.6}`.
2. **`test_static_frontend_serving(client)`:**
   - Verify `GET /`, `GET /style.css`, `GET /app.js`, `GET /logo.svg`, and `GET /static/mock_result.json` all return HTTP `200`.
   - Verify `GET /` HTML contains `<meta charset="utf-8">` and `DermaScan Face AI`.
3. **`test_valid_selfie_and_contract_keys(client)`:**
   - Select a test-split full-face selfie from `manifest.csv` (`split == "test"` and `face_detected == True`).
   - `POST /api/analyze` returns `200`.
   - Verify all 21 frozen contract keys exist (`prediction`, `prediction_label`, `confidence`, `abstained`, `abstention_threshold`, `abstention_message`, `caution_note`, `triage_mode`, `probabilities`, `top2`, `input_mode`, `face_detected`, `num_patches`, `dominant_region`, `region_probabilities`, `model_breakdown`, `original_image`, `gradcam_image`, `latency_ms`, `info`, `disclaimer`).
   - Verify `input_mode == "face"`, `face_detected is True`, `num_patches == 5`, `set(region_probabilities.keys()) == {"forehead", "nose", "left_cheek", "right_cheek", "chin"}`.
   - Verify `abs(sum(probabilities.values()) - 1.0) < 1e-3`.
   - Verify `original_image` and `gradcam_image` start with `"data:image/png;base64,"` and decode to valid non-empty PNG bytes.
   - Verify `info` contains `"en"`, `"th"`, and `"sources"` with Thai text present in `info["th"]["name"]`.
4. **`test_valid_closeup_and_triage_mode(client)`:**
   - Select a test-split close-up image from `manifest.csv` (`split == "test"` and `face_detected == False`).
   - `POST /api/analyze` with `triage_mode="true"` returns `200`.
   - Verify `input_mode == "closeup"`, `face_detected is False`, `1 <= num_patches <= 9`, `region_probabilities is None`.
   - Verify `triage_mode is True` and `probabilities["normal"] == 0.0`.
5. **`test_all_invalid_inputs_and_error_states(client)`:**
   - **No file provided (`POST /api/analyze` with empty data):** returns `400` with `error_code == "empty_file"`.
   - **Empty 0-byte file (`empty.jpg`):** returns `400` with `error_code == "empty_file"`.
   - **Unsupported file extension (`notes.txt` or `doc.pdf`):** returns `415` with `error_code == "invalid_file_type"`.
   - **Corrupted image bytes (`corrupt.jpg` with `b"not-a-valid-jpeg"`):** returns `400` with `error_code == "unreadable_image"`.
   - **Oversized file (`big.jpg` with `MAX_UPLOAD_BYTES + 100` bytes):** returns `413` with `error_code == "file_too_large"`.
   - **No face & no usable skin region (Case 8: a 300x300 solid black or blue/green non-skin PNG encoded in memory via `cv2.imencode`):** returns `422` with `error_code == "no_skin_detected"`, plus non-empty `detail` and `detail_th`.
6. **`test_consistency_against_test_predictions_csv(client)`:**
   - Load `runs/ensemble_seed42/test_predictions.csv` and merge with `data/manifest/manifest.csv` on `image_id` to get `path` and `face_detected`.
   - Compute calibrated probabilities `cal_probs` for all 303 rows using `predict_calibrated_ensemble(res_probs, eff_probs, w_resnet=0.40, t_resnet=0.60, t_effnet=1.20)`.
   - Select **16 representative test-split images**: 4 correctly classified high-confidence images per class (`Acne`, `Rosacea`, `Dermatitis`, `Normal`).
   - For each of the 16 images, read its raw bytes from `PROJECT_ROOT / row["path"]` and call `POST /api/analyze`.
   - Compute the per-image max absolute probability difference between the API's `probabilities` and `cal_probs[i]`.
   - Assert that:
     - At least **15/16 (`>= 93.75%`)** match the exact top-1 predicted class from `cal_probs` (accounting for any borderline image affected by JPEG-95 patch saving vs lossless in-memory cropping).
     - The mean absolute probability difference across all 16 images is **`< 0.03`** (`3%`).
7. **`test_eight_demo_cases_and_benchmark_summary(client)`:**
   - Programmatically identify and test concrete test-split images from `test_predictions.csv` + `manifest.csv` for each of the **8 required demo cases** from Section 6 of `WEBAPP_HANDOFF_PROMPT.md`:
     1. **Case 1 (Full-face selfie -> Acne):** `split == "test"`, `face_detected == True`, true & predicted `Acne`, not abstained.
     2. **Case 2 (Full-face selfie -> Rosacea):** `split == "test"`, `face_detected == True`, true & predicted `Rosacea`, not abstained.
     3. **Case 3 (Dermatitis case):** `split == "test"`, true & predicted `Dermatitis`, not abstained (verify `caution_note` is present!).
     4. **Case 4 (Normal face):** `split == "test"`, `face_detected == True`, true & predicted `Normal`, not abstained (verify `caution_note` is present!).
     5. **Case 5 (Close-up, no face detected -> tiled patches path):** `split == "test"`, `face_detected == False`, `input_mode == "closeup"`, correctly classified.
     6. **Case 6 (Low-confidence image -> abstention banner):** Pick a test-split image where calibrated max confidence `< ABSTENTION_THRESHOLD (0.7852)` and verify `POST /api/analyze` returns `abstained is True` and non-empty `abstention_message` with `top2`.
     7. **Case 7 (Invalid input: non-image / corrupted / too large):** Verify HTTP `415` / `400` / `413`.
     8. **Case 8 (Image with no face and no usable skin region):** Verify HTTP `422` (`error_code == "no_skin_detected"`).
   - Print a clean summary block to stdout showing:
     - The exact `image_id`, relative `path`, `input_mode`, `num_patches`, `prediction`, `confidence`, and `latency_ms` for Demo Cases 1–6.
     - Mean, median, min, and max end-to-end latency (`ms`) across the tested images, and peak GPU VRAM allocated (`torch.cuda.max_memory_allocated() / (1024**2)` MB).

### Step 2: Run `pytest`
Run `pytest tests/test_webapp.py -v -s` from `e:\CPE310FacialDermatorsisProject` and capture the full output for the handoff report.

## 6. Hard Rules
- **DO NOT** modify `webapp/*`, `src/*`, `data/raw/`, `checkpoints/`, or `.env`.
- Run all commands from `e:\CPE310FacialDermatorsisProject` (no `cd`). Use UTF-8 encoding.
- Only use images from the **in-domain `test` split** for demo cases 1–6 (never training images).

## 7. Acceptance Criteria
Run this command from `e:\CPE310FacialDermatorsisProject` and paste its complete output into your handoff report:
```powershell
$env:PYTHONIOENCODING='utf-8'; pytest tests/test_webapp.py -v -s
```
Expected: All 7 test functions in `tests/test_webapp.py` pass (`7 passed`), the 16-image consistency check passes (`mean_abs_diff < 0.03`), and the printed summary lists the exact test-split image paths for Demo Cases 1–6 plus latency and peak VRAM numbers.

## 8. Stop Conditions
- If any test fails due to a bug in `webapp/service.py` or `webapp/app.py`, **STOP** and report the exact failure traceback in `tasks/task_w5_handoff_report.md`. Do not edit `webapp/service.py` or `webapp/app.py` yourself.

## 9. Report Instructions
When finished, write `tasks/task_w5_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact `pytest` command run and its actual output (pasted verbatim)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check
- **Numbers to carry forward:**
  1. 16-image consistency check results (top-1 agreement fraction and mean/max absolute probability difference).
  2. Exact `image_id`, file `path`, `input_mode`, `num_patches`, `prediction`, `confidence`, and `top2` for Demo Cases 1–6 (so Task W6 can put them directly into `DEMO_SCRIPT.md`).
  3. End-to-end latency (`mean`, `median`, `min`, `max` in ms) and Peak GPU VRAM (MB).

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w5_tests_and_benchmarks.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** `pytest tests/test_webapp.py -v -s` printed `7 passed` and output the table of Demo Cases 1–6 with real test-split image paths, latency, and VRAM.
