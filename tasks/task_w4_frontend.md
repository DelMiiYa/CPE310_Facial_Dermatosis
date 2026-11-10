# Goldfish Task Spec: W4 — Responsive Frontend UI & Mock Result (`webapp/static/*`)

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier (Acne, Rosacea, Dermatitis, Normal) for a university Healthcare AI System course in Thailand.
The app will be demoed live to the professor and recorded in a video clip (which must start with the system name and logo and cover all 8 demo cases).
Task W4 builds the complete, responsive vanilla HTML/CSS/JS frontend in `webapp/static/` against a frozen API contract and a local `webapp/static/mock_result.json` file so it works immediately and connects seamlessly to `POST /api/analyze`.

## 2. Goal
Create `webapp/static/logo.svg`, `webapp/static/mock_result.json`, `webapp/static/index.html`, `webapp/static/style.css`, and `webapp/static/app.js` implementing the complete bilingual (EN/TH) diagnostic interface against the frozen `/api/analyze` contract.

## 3. Files to Read First
Read these exact files before writing any code:
- `tasks/WEBAPP_MASTER_PLAN.md` (Section 4.1, 4.2, 4.3, 4.4: Frozen API & Knowledge Base Contracts)
- `Project_handoff_summary.md` (Section 3 Steps 9–13 for the real project metrics to display in the "Model Architecture & Limitations" panel)

### Copied Frozen `/api/analyze` Response Contract from `tasks/WEBAPP_MASTER_PLAN.md`
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
  "probabilities": {
    "acne": 0.0412,
    "rosacea": 0.9124,
    "dermatitis": 0.0281,
    "normal": 0.0183
  },
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
    "calibration_params": {
      "t_resnet": 0.6,
      "t_effnet": 1.2,
      "w_resnet": 0.4,
      "w_effnet": 0.6
    }
  },
  "original_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,...",
  "latency_ms": 124.5,
  "info": {
    "name": "Rosacea",
    "model_match_phrase": "The model's best match is Rosacea.",
    "short_description": "...",
    "typical_symptoms": ["..."],
    "typical_facial_distribution": "...",
    "common_triggers_or_factors": ["..."],
    "commonly_confused_with": [{"condition": "Acne Vulgaris", "how_it_differs": "..."}],
    "general_care_tips": ["..."],
    "see_a_doctor_if": ["..."],
    "caution_note": null,
    "en": { ...same 10 keys... },
    "th": { ...same 10 keys in Thai... },
    "sources": [{"title": "...", "url": "..."}]
  },
  "disclaimer": "Educational research demonstration only — NOT a medical diagnosis. Do not use for clinical decision-making. Please consult a qualified dermatologist."
}
```
And for error responses (`400`, `413`, `415`, `422`, `500`):
```json
{
  "detail": "Clear human-readable message in English",
  "detail_th": "ข้อความแจ้งเตือนภาษาไทย",
  "error_code": "invalid_file_type | file_too_large | empty_file | unreadable_image | no_skin_detected | server_error"
}
```

## 4. Files to Create or Edit
Create ONLY these files (everything else is strictly off limits):
1. `webapp/static/logo.svg`
2. `webapp/static/mock_result.json`
3. `webapp/static/index.html`
4. `webapp/static/style.css`
5. `webapp/static/app.js`

## 5. Step-by-Step Instructions

### Step 1: Create `webapp/static/logo.svg`
Create a clean, modern medical-AI SVG icon (`64x64` viewBox) depicting a stylized face outline with 5 anatomical scan zones (forehead, nose, cheeks, chin) and a clinical cross/shield accent in teal/indigo (`#0d9488` / `#1e40af`).

### Step 2: Create `webapp/static/mock_result.json`
Create a complete, valid JSON file matching the frozen `/api/analyze` response contract above (for `"rosacea"` in `"face"` mode with 5 `region_probabilities`, `model_breakdown`, realistic `info` with both `en` and `th` objects and `sources`, and valid 1x1 or small synthetic PNG data URLs in `original_image` and `gradcam_image`) so that opening `index.html?mock=1` or clicking "Preview Demo Result (Mock)" renders the entire UI without needing the backend running.

### Step 3: Create `webapp/static/index.html`
Build a clean, semantic HTML5 single-page application:
1. `<meta charset="utf-8">` and `<meta name="viewport" content="width=device-width, initial-scale=1.0">`.
2. Load Google Fonts for `Inter` and `Sarabun` / `Noto Sans Thai` so both English and Thai render crisply.
3. **Header:**
   - Displays `logo.svg`, system title (`<span id="system-name">DermaScan Face AI</span>`), subtitle (`CPE310 Healthcare AI System — Calibrated Dual-Backbone Facial Dermatosis Classifier`), and a **Language Toggle button (`EN | ไทย`)** that switches all UI labels and the Condition Knowledge Base Card between English and Thai immediately without re-running inference.
4. **Always-Visible Medical & Privacy Banner:**
   - States clearly that this is an educational university demonstration, NOT a clinical diagnostic device, and that **uploaded photos are processed strictly in memory and never saved to disk**.
5. **Upload & Options Panel (Left column on desktop, top on mobile):**
   - Drag-and-drop zone + file picker (`<input type="file" id="file-input" accept="image/jpeg,image/png,image/webp" capture="user">`) noting supported formats (`JPG, PNG, WEBP`, max `10 MB`).
   - Client-side preview of the selected photo + filename + file size, with a "Clear / Change Photo" button.
   - **Symptomatic Triage Mode checkbox (`id="triage-mode-toggle"`):** Optional toggle ("Symptomatic 3-Class Triage Mode — excludes Normal class via Bayesian prior adjustment for known lesions").
   - **"Analyze Photo" primary button (`id="analyze-btn"`):** Disabled when no file is selected or while a request is in flight; shows a loading spinner and "Analyzing facial patches & Grad-CAM..." state during requests.
   - **"Load Demo Preview (Mock)" secondary button (`id="mock-btn"`):** Fetches `/static/mock_result.json` (or `mock_result.json` relative path) and renders the full result view for offline UI testing.
   - **Friendly Error Alert Box (`id="error-banner"`, hidden by default):** Displays `detail` (or `detail_th` when language is Thai) and `error_code` badge with clear guidance when an upload is rejected (invalid file type, >10 MB, corrupted image, or no face/usable skin region detected).
6. **Results Panel (`id="results-panel"`, hidden until analysis completes):**
   - **Abstention Banner (`id="abstention-banner"`, shown when `abstained === true`):** High-visibility amber/warning banner stating `"Uncertain — Please Consult a Dermatologist"` (and Thai translation), showing `abstention_message` and highlighting the **Top-2 candidate conditions** instead of presenting a hard diagnosis.
   - **Primary Prediction Card:**
     - Shows `"The model's best match is:"` (or `"Uncertain — Top candidate:"` when abstained), the predicted condition badge (`Acne` / `Rosacea` / `Dermatitis` / `Normal`), confidence percentage, latency badge (`latency_ms` ms), and input mode badge:
       - If `input_mode === "face"`: `"Full-face selfie detected: 5 anatomical regions analysed (forehead, nose, left cheek, right cheek, chin) • Dominant region: <dominant_region>"`
       - If `input_mode === "closeup"`: `"Close-up macro image: <num_patches> patches analysed • Dominant patch: <dominant_region>"`
     - **Caution Note Callout (`id="caution-note-box"`, shown when `caution_note` is non-null):** Displays the cautious wording for `Dermatitis` (smallest training class, F1 0.62–0.71, merges atopic + contact dermatitis) or `Normal` (absence of target 3 diseases, not a guarantee of health).
   - **4-Class Probability Bars (`id="probability-bars"`):**
     - Horizontal progress bars + numeric percentages for all 4 classes (`Acne`, `Rosacea`, `Dermatitis`, `Normal`), sorted highest-first, with distinct accessible patterns/badges as well as colour, and a visual marker indicating the validation abstention threshold (`abstention_threshold`).
   - **Visual Explainability (Grad-CAM Overlay & Region Breakdown):**
     - Image viewer with segmented toggle buttons: **`Show Grad-CAM Heatmap`**, **`Show Original Image`**, and **`Side-by-Side`**.
     - **Per-Region Probability Table (`id="region-table-container"`, shown when `input_mode === "face"` and `region_probabilities` is present):** A clean 5-row table (`Forehead`, `Nose`, `Left Cheek`, `Right Cheek`, `Chin`) showing the top predicted class and all 4 class probabilities per facial region, highlighting the `dominant_region` row.
   - **Collapsible Model Ensemble Breakdown (`id="model-breakdown-card"`):**
     - Shows side-by-side comparison of **ResNet-50** ($T=0.60, w=0.40$), **EfficientNet-B0** ($T=1.20, w=0.60$), and **Calibrated Ensemble**, plus an Agreement badge (`Models Agree` vs `Models Disagree — Resolved by Calibrated Soft Fusion`).
   - **Condition Educational Information Card (`id="condition-info-card"`):**
     - Renders `info.en` or `info.th` based on the active language toggle:
       1. `model_match_phrase` & `short_description`
       2. `typical_symptoms` (bullet list)
       3. `typical_facial_distribution` (highlighted box linking clinical localization to Grad-CAM)
       4. `common_triggers_or_factors` (bullet list)
       5. `commonly_confused_with` (differential table/list showing `condition` and `how_it_differs`)
       6. `general_care_tips` (conservative self-care list)
       7. `see_a_doctor_if` (red-flag warning list)
       8. `sources` (clickable reference links with `target="_blank" rel="noopener noreferrer"`)
7. **Expandable "About This Model & Known Limitations" Panel (`<details id="limitations-panel">`):**
   - Must display the **real, honest benchmark numbers** from `Project_handoff_summary.md`:
     - **Architecture:** MediaPipe Face Detector & 478-landmark 5-region patch extractor + Calibrated Dual-Backbone Ensemble (`ResNet-50` 23.52M params + `EfficientNet-B0` 4.01M params; validation-fit temperature scaling $T_{\text{resnet}}=0.60, T_{\text{effnet}}=1.20, w_{\text{resnet}}=0.40$).
     - **Dataset & Split:** 2,214 curated smartphone images across 1,491 subjects (70/15/15 subject-grouped stratified split: 1,617 Train / 294 Val / 303 In-Domain Test) + 337 held-out Google SCIN external test images.
     - **In-Domain Test Performance (303 images):** **92.74% Accuracy**, **0.8736 Macro-F1**, **0.0197 ECE** (Per-class F1: Normal `0.9804`, Acne `0.9412–0.9464`, Rosacea `0.9091–0.9100`, Dermatitis `0.6207–0.7059`).
     - **Selective Abstention Safety:** Withholding low-confidence predictions achieves **97.07% Accuracy at 90% coverage** and **99.17% Accuracy (240/242) at 80% coverage** on the in-domain test set.
     - **Known Limitations:**
       1. *Dermatitis Sample Size:* Only 93 training images (15 val, 15 test); merges atopic eczema and contact dermatitis.
       2. *External SCIN Smartphone Domain Gap:* Raw 4-class accuracy on the held-out SCIN benchmark (337 images, 82.5% macro close-ups, 72.7% Dermatitis, 0% Normal) is **20.77%** (Macro-F1 `0.1665`), which rises to **44.51% Accuracy** (Macro-F1 `0.3321`, Dermatitis recall `45.31%`) under 3-class Symptomatic Triage Mode.
       3. *4-Class Scope:* The model only recognizes Acne, Rosacea, Dermatitis, and Normal; it cannot identify skin cancers, infections, or pigmentary disorders.

### Step 4: Create `webapp/static/style.css` and `webapp/static/app.js`
1. In `webapp/static/style.css`:
   - Responsive two-column grid on desktop (`>= 960px`), single-column stack on mobile/tablet (`< 960px`).
   - High-contrast clinical colour palette (WCAG AA compliant), clear focus states, smooth progress bar transitions, print/clip-friendly typography (`Inter`, `'Sarabun'`, `'Noto Sans Thai'`, sans-serif).
2. In `webapp/static/app.js`:
   - Define `const SYSTEM_NAME = "DermaScan Face AI";` at the top of the file.
   - Perform client-side validation before `fetch("/api/analyze")` (check file extension/MIME type and `file.size <= 10 * 1024 * 1024`; if invalid, display a friendly bilingual message in `#error-banner` immediately).
   - Send `POST /api/analyze` with `FormData` (`file` and `triage_mode`).
   - Handle non-2xx JSON responses (`detail`, `detail_th`, `error_code`) gracefully in `#error-banner`.
   - Support `?mock=1` URL parameter or clicking `#mock-btn` to load `mock_result.json` (trying `/static/mock_result.json` then `./mock_result.json`) and call `renderResult(data)`.
   - Implement `setLanguage(lang)` (`"en"` or `"th"`) that re-renders all static labels and the active `lastResult.info[lang]` card immediately.

## 6. Hard Rules
- **No build step, no npm, no external JS frameworks** (vanilla HTML5, CSS3, ES6 JavaScript only).
- **UTF-8 encoding is mandatory** for all files so Thai text renders properly.
- **DO NOT** touch `data/raw/`, `checkpoints/`, `.env`, or `src/`.
- **Do not embed restricted clinical/DermNet images** in `webapp/static/`.

## 7. Acceptance Criteria
Run this verification command from `e:\CPE310FacialDermatorsisProject` and paste its exact output into your handoff report:
```powershell
python -c "import json; from pathlib import Path; files = ['logo.svg', 'mock_result.json', 'index.html', 'style.css', 'app.js']; [print(f, 'exists:', (Path('webapp/static')/f).exists(), 'bytes:', (Path('webapp/static')/f).stat().st_size) for f in files]; m = json.loads(Path('webapp/static/mock_result.json').read_text(encoding='utf-8')); req_keys = ['prediction', 'prediction_label', 'confidence', 'abstained', 'abstention_threshold', 'probabilities', 'top2', 'input_mode', 'num_patches', 'region_probabilities', 'model_breakdown', 'original_image', 'gradcam_image', 'latency_ms', 'info', 'disclaimer']; assert all(k in m for k in req_keys), 'Missing mock keys'; html = Path('webapp/static/index.html').read_text(encoding='utf-8'); assert 'utf-8' in html.lower() and 'DermaScan Face AI' in html, 'Missing meta or title'; print('FRONTEND STATIC & MOCK CONTRACT CHECKS PASSED')"
```
Expected: All 5 files exist with non-zero byte sizes and prints `FRONTEND STATIC & MOCK CONTRACT CHECKS PASSED`.

## 8. Stop Conditions
- If `tasks/WEBAPP_MASTER_PLAN.md` is missing, **STOP** and report that in `tasks/task_w4_handoff_report.md`.

## 9. Report Instructions
When finished, write `tasks/task_w4_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact commands run and their actual output (pasted verbatim)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check
- **Numbers to carry forward:** file sizes and DOM IDs/endpoints used (`POST /api/analyze`, `FormData` fields `file` and `triage_mode`).

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session 3)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w4_frontend.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** The acceptance command printed `FRONTEND STATIC & MOCK CONTRACT CHECKS PASSED` and all 5 files in `webapp/static/` exist.
