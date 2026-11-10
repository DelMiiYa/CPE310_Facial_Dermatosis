# Task W7 — Elephant Final Audit Report (`DermaScan Face AI`)

**Date:** 2026-10-24  
**Auditor:** Elephant (Planner / Architect Agent)  
**Project:** CPE310 Facial Dermatosis Classifier — Demo Web Application (`webapp/`)  
**Overall Audit Verdict:** **PASS (100% of W0–W7 deliverables, 8 demo cases, and hard rules verified)**

---

## 1. Deliverable Inventory & Status Verification

| Task ID | Deliverable Path | Verification Status | Key Evidence |
|---|---|---|---|
| **W0** | `tasks/WEBAPP_MASTER_PLAN.md` | **PASS** | Frozen `/api/analyze` (`200 OK`), `/api/health`, error schema, and bilingual KB contracts documented; all tasks W0–W7 marked **DONE**. |
| **W1** | `scripts/pick_abstention_threshold.py` | **PASS** | Derived `ABSTENTION_THRESHOLD = 0.7852` strictly from `runs/ensemble_seed42/val_predictions.csv` (294 val images) at 85.0% coverage (`250/294`, `Val Acc = 99.20%`, `Val Macro-F1 = 0.9940`). Never touched `test` or `scin` splits. |
| **W1** | `webapp/__init__.py` & `webapp/service.py` | **PASS** | Implements `WebAppDermatosisService`, `analyze()`, `has_usable_skin_region()`, `WebAppInputError`, single-copy dual-backbone Grad-CAM sharing (`self.ensemble.resnet` + `self.ensemble.effnet`), and thread-safe GPU inference lock (`self._gpu_lock`). |
| **W2** | `webapp/knowledge_base.json` | **PASS** | Valid UTF-8 JSON with all 4 classes (`acne`, `rosacea`, `dermatitis`, `normal`), top-level English keys, full `en` and `th` localized sub-objects, and 16 reputable medical URLs (AAD, NRS, NHS, Mayo Clinic, NEA, DermNet NZ). No prescription drug doses. |
| **W3** | `webapp/app.py` & `requirements.txt` | **PASS** | FastAPI app with `lifespan` singleton initialization, `GET /api/health`, `POST /api/analyze`, static asset mounts (`/`, `/style.css`, `/app.js`, `/logo.svg`, `/static/*`), and structured JSON error handling (`415`, `400`, `413`, `422`). |
| **W4** | `webapp/static/{index.html,style.css,app.js,mock_result.json,logo.svg}` | **PASS** | Responsive clinical dashboard with instant client-side EN/TH toggle, drag-and-drop + camera capture, 4-class probability bars with 78.5% threshold marker, Grad-CAM heatmap toggle, 5-region breakdown table, ResNet-50 vs. EfficientNet-B0 comparison cards, and permanent limitations & disclaimer panels. |
| **W5** | `tests/test_webapp.py` | **PASS** | `pytest tests/test_webapp.py -v` passes **7/7** test suites in `33.75s`. Includes 16-image consistency check against `runs/ensemble_seed42/test_predictions.csv` (`16/16 = 100.0%` top-1 agreement, mean max absolute probability difference `0.0028 < 0.03`). |
| **W6** | `WEBAPP_README.md` & `DEMO_SCRIPT.md` | **PASS** | Complete setup/launch guide, ASCII + Mermaid architecture diagrams, frozen JSON contracts, model metrics, and a step-by-step 8-case recording script with real test-split image paths and bilingual talking points. |
| **W7** | `tasks/webapp_final_audit.md` | **PASS** | This final audit report. |

---

## 2. Verification of the 8 Required Demo Cases (`WEBAPP_HANDOFF_PROMPT.md` Section 7)

All 8 demo cases were verified end-to-end against real `test`-split images via `POST /api/analyze` (in `tests/test_webapp.py::test_eight_demo_cases_and_benchmark_summary`) and documented in `DEMO_SCRIPT.md`:

| Case # | Scenario | Real Test Image / Input Path | Observed API Result | Status |
|---|---|---|---|---|
| **Case 1** | Clear Frontal Face — **Acne** | `data/raw/DermNet_Grab/test/acne/07Acne0811011.jpg` | `input_mode="face"`, `face_detected=True`, `num_patches=5`, `prediction="acne"`, `confidence=0.9037` (`90.4%`), `abstained=False`, `dominant_region="chin"` | **PASS** |
| **Case 2** | Clear Frontal Face — **Rosacea** | `data/raw/DermNet_Grab/test/rosacea/07PerioralDermEye.jpg` | `input_mode="face"`, `face_detected=True`, `num_patches=5`, `prediction="rosacea"`, `confidence=0.9614` (`96.1%`), `abstained=False`, `dominant_region="right_cheek"` | **PASS** |
| **Case 3** | Close-Up / Macro — **Dermatitis** | `data/raw/DermNet_Grab/test/dermatitis/03ichthyosis050127.jpg` | `input_mode="closeup"`, `face_detected=False`, `num_patches=2`, `prediction="dermatitis"`, `confidence=0.8099` (`81.0%`), `abstained=False`, `caution_note` populated | **PASS** |
| **Case 4** | Clear Frontal Face — **Normal Skin** | `data/raw/celeba_normal/000015.jpg` | `input_mode="face"`, `face_detected=True`, `num_patches=5`, `prediction="normal"`, `confidence=0.9963` (`99.6%`), `abstained=False`, `dominant_region="nose"` | **PASS** |
| **Case 5** | Close-Up Macro Fallback Path | `data/raw/DermNet_Grab/test/acne/07AcnePittedScars.jpg` | `input_mode="closeup"`, `face_detected=False`, `num_patches=2`, `prediction="acne"`, `confidence=0.9457` (`94.6%`), `abstained=False` | **PASS** |
| **Case 6** | Ambiguous / Low-Confidence (**Abstention**) | `data/raw/DermNet_Grab/test/dermatitis/05KeratosisPilaris1.jpg` (`triage_mode=false`) | `confidence=0.5171 < 0.7852`, `abstained=True`, `top2=["rosacea", "dermatitis"]`, amber abstention warning banner rendered | **PASS** |
| **Case 7** | **Triage Mode** (`triage_mode=true`) | `data/raw/DermNet_Grab/test/dermatitis/03EczemaStaph011904.jpg` (`triage_mode=true`) | `confidence=0.7745 < 0.7852`, `abstained=False` (bypassed by `triage_mode=true`), `caution_note` warns that confidence is below standard threshold | **PASS** |
| **Case 8** | Invalid / Non-Skin Rejection (`415` / `400` / `413` / `422`) | `data/reports/demo_invalid_inputs/{not_an_image.txt,corrupted_photo.jpg,non_skin_blue.png}` | Returns structured JSON error (`unsupported_media_type` 415, `corrupted_image` 400, `file_too_large` 413, `no_skin_detected` 422) without crashing | **PASS** |

---

## 3. Performance, Latency & Hardware Budget Audit (NVIDIA RTX 3050 4 GB Laptop GPU)

- **Startup & Model Sharing:** `WebAppDermatosisService` loads `DermatosisEnsemble` (`runs/resnet50_seed42/best_model.pth`, `runs/effnet_b0_seed42/best_model.pth`, `runs/ensemble_seed42/temperature.pt`) once during FastAPI lifespan startup and attaches `GradCAM` hooks directly to `self.ensemble.resnet` (`layer4`) and `self.ensemble.effnet` (`conv_head`).
- **VRAM Usage:**
  - Single-request peak GPU VRAM: **`1130.1 MB`** (`1.10 GB`).
  - Full 25+ image automated test suite peak GPU VRAM: **`2514.3 MB`** (`2.46 GB`), well inside the `< 3800 MB` budget on the 4 GB RTX 3050.
- **Inference Latency:**
  - Steady-state end-to-end latency (including MediaPipe face/landmark detection, 5-patch extraction, dual-backbone inference, and dual-backbone Grad-CAM overlay generation): **`125.5 ms – 205.8 ms`** (median **`179.4 ms`**, mean **`362.7 ms`** including initial warmup).

---

## 4. Compliance with Hard Project Rules (`WEBAPP_HANDOFF_PROMPT.md` Section 8)

Verified via `git status -s`:
1. **`data/raw/`, `checkpoints/`, and `.env` untouched:** Confirmed (`git status -s` shows zero changes in `data/raw/`, `checkpoints/`, or `.env`).
2. **Existing `src/*.py` and `data/reports/step*.md` untouched:** Confirmed (`git status -s` shows zero modifications to `src/` or `data/reports/step*.md`).
3. **No model retraining or `test`/`scin` tuning:** Confirmed (`ABSTENTION_THRESHOLD = 0.7852` was selected strictly from `runs/ensemble_seed42/val_predictions.csv`).
4. **Medical Safety & Bilingual Framing:** Confirmed (`webapp/knowledge_base.json` and `webapp/static/index.html` use educational match framing, include prominent EN/TH non-diagnostic disclaimers, cite 16 reputable medical URLs, and contain no prescription drug dosages).
