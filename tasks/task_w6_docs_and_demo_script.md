# Goldfish Task Spec: W6 — Documentation (`WEBAPP_README.md` & `DEMO_SCRIPT.md`)

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier web application (`DermaScan Face AI`).
Tasks W1–W5 are complete and verified (`7 passed` in `tests/test_webapp.py`, 16/16 consistency check with `0.0028` mean probability diff, `179.4 ms` median latency, and `2514.3 MB` peak VRAM on RTX 3050).
Task W6 writes the two comprehensive documentation deliverables required for the assignment and live demo clip: `WEBAPP_README.md` and `DEMO_SCRIPT.md`.

## 2. Goal
Create `WEBAPP_README.md` (setup, one-command run instructions, architecture block diagram for the clip, validation abstention derivation, latency/VRAM benchmarks, limitations, and medical review flag) and `DEMO_SCRIPT.md` (complete 8-case live demo & video recording script with verified in-domain test-split image paths and presenter talking points).

## 3. Files to Read First
Read these exact files before writing:
- `tasks/WEBAPP_MASTER_PLAN.md`
- `tasks/task_w1_handoff_report.md` (validation abstention table & single-request VRAM)
- `tasks/task_w2_handoff_report.md` (16 medical reference URLs)
- `tasks/task_w3_handoff_report.md` (HTTP status codes & TestClient latency)
- `tasks/task_w5_handoff_report.md` (16-image consistency results, Demo Cases 1–6 metadata, latency & peak VRAM)
- `Project_handoff_summary.md` (Steps 9–13 metrics)

### Copied Verified Numbers from Tasks W1–W5
1. **Run Command (from project root `e:\CPE310FacialDermatorsisProject`):**
   ```powershell
   python -m uvicorn webapp.app:app --port 8000
   ```
   Open `http://localhost:8000` in browser (or `http://localhost:8000/?mock=1` for offline mock preview).
2. **Test Command:**
   ```powershell
   $env:PYTHONIOENCODING='utf-8'; pytest tests/test_webapp.py -v -s
   ```
3. **Validation Selective Prediction Table (`scripts/pick_abstention_threshold.py` on 294 Validation Images):**
   | Coverage (%) | Retained (`k/total`) | Abstention Count | Threshold ($\tau$) | Val Accuracy (%) | Val Macro-F1 |
   |---|---|---|---|---|---|
   | 100.0% | 294/294 | 0 | `0.2971` | 96.26% | 0.9372 |
   | 95.0% | 279/294 | 15 | `0.5493` | 97.13% | 0.9498 |
   | 90.0% | 265/294 | 29 | `0.6798` | 98.49% | 0.9705 |
   | **85.0% (Selected)** | **250/294** | **44** | **`0.7852`** | **99.20%** | **0.9940** |
   | 80.0% | 235/294 | 59 | `0.8339` | 99.15% | 0.9937 |
   | 75.0% | 220/294 | 74 | `0.8692` | 99.55% | 0.9967 |
   | 70.0% | 206/294 | 88 | `0.9257` | 99.51% | 0.9966 |
   *(Reference on In-Domain Test Set [303 images]: 100% coverage = 92.74% Acc / 0.8736 Macro-F1; 90% coverage = 97.07% Acc / 0.9320 Macro-F1; 80% coverage = 99.17% Acc [240/242 correct] / 0.9464 Macro-F1).*
4. **Latency & GPU Memory Benchmarks (NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM):**
   - **Steady-State End-to-End Request Latency (upload -> face/tile extraction -> dual-backbone inference -> calibration -> dual Grad-CAM -> PNG base64 -> JSON):**
     - **Median:** `179.4 ms`
     - **Min:** `125.5 ms` (1-patch close-up)
     - **Typical 5-Patch Selfie:** `154.0 – 205.8 ms`
     - **Mean (including initial warmup pass of 1331.9 ms):** `362.7 ms`
   - **GPU VRAM Footprint:**
     - **Single-Request Model + Grad-CAM Allocation:** `1130.1 MB`
     - **Peak VRAM Across Full 25+ Image Pytest Suite:** `2514.3 MB` (comfortably below the 4096 MB hardware limit).
   - **Consistency vs. `runs/ensemble_seed42/test_predictions.csv` (16 Test Images):**
     - **Top-1 Class Agreement:** `16 / 16` (`100.00%`)
     - **Mean Max Absolute Probability Difference:** `0.0028` (`0.28%`, max `0.0207` due to lossless in-memory patch cropping vs. JPEG-95 pre-extracted patches on disk).
5. **Verified In-Domain Test-Split Images for Demo Cases 1–6 (`split == "test"` in `data/manifest/manifest.csv`):**
   | Case | Scenario | `image_id` | Exact File Path (from project root) | Input Mode | Patches | Prediction | Confidence | Top-2 | Abstained |
   |---|---|---|---|---|---|---|---|---|---|
   | **Case 1** | Full-face selfie -> Acne | `iga34_65760604461e` | `data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg` | `face` | 5 | `acne` | `100.00%` | `['acne', 'rosacea']` | `False` |
   | **Case 2** | Full-face selfie -> Rosacea | `roboflow_rosacea_030fa3e083ad` | `data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg` | `face` | 5 | `rosacea` | `99.99%` | `['rosacea', 'dermatitis']` | `False` |
   | **Case 3** | Dermatitis case (shows caution note) | `roboflow_face_skin_f9ef0ad817a8` | `data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg` | `closeup` | 6 | `dermatitis` | `99.15%` | `['dermatitis', 'rosacea']` | `False` |
   | **Case 4** | Normal face (shows caution note) | `skin_assessment_fba55cb1f7a8` | `data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg` | `face` | 5 | `normal` | `100.00%` | `['normal', 'dermatitis']` | `False` |
   | **Case 5** | Close-up (no face detected -> tiled patches) | `roboflow_rosacea_c2e7137b2c96` | `data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg` | `closeup` | 1 | `rosacea` | `100.00%` | `['rosacea', 'acne']` | `False` |
   | **Case 6** | Low-confidence image -> Abstention banner | `roboflow_face_skin_4b730dec0c6d` | `data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg` | `face` | 5 | `rosacea` | `40.48%` | `['rosacea', 'normal']` | `True` |

   *(Note to include in `DEMO_SCRIPT.md`: All 6 images above are strictly in the held-out in-domain `test` split [`split == "test"` in `data/manifest/manifest.csv`]. Some raw files reside in a folder named `.../train/...` only because Roboflow's original download zip used that folder name; our pipeline ignored Roboflow's folders, deduplicated via 64-bit pHash, and assigned subject-grouped 70/15/15 splits in `manifest.csv`).*

## 4. Files to Create or Edit
Create ONLY these two files (everything else is strictly off limits):
1. `WEBAPP_README.md`
2. `DEMO_SCRIPT.md`

## 5. Step-by-Step Instructions

### Step 1: Write `WEBAPP_README.md`
Include these sections in `WEBAPP_README.md`:
1. **System Overview & Naming (`DermaScan Face AI`):**
   - Explain what the web app does and how to customize the system name (`SYSTEM_NAME` in `webapp/service.py` and `webapp/static/app.js`) or logo (`webapp/static/logo.svg`). Provide 3–4 alternative name ideas (e.g. *DermaScan Face AI*, *SkinSight CPE310*, *ThaiDerma Lens*, *FaceDerma Explain*).
2. **Quick Start (One-Command Launch & Testing):**
   - Exact command to launch the server (`python -m uvicorn webapp.app:app --port 8000`), URL (`http://localhost:8000`), offline mock preview URL (`http://localhost:8000/?mock=1`), and `pytest` command.
3. **Architecture Block Diagram (for Presentation Clip & Report):**
   - Include a Mermaid `flowchart TD` diagram AND a clear numbered block-by-block walkthrough mapping each processing stage to its exact Python module/function:
     `Upload (webapp/app.py) -> Face Detection & 478-Landmark Extraction (src/detection.py) -> 5 Anatomical Regions or 1–9 Close-Up Tiles (with YCrCb/HSV usable-skin validation) -> Dual-Backbone Forward Pass (ResNet-50 + EfficientNet-B0 in src/ensemble.py) -> Validation-Fit Temperature Scaling & Soft Fusion (T_resnet=0.60, T_effnet=1.20, w=0.40/0.60 in src/calibration.py) -> Optional 3-Class Symptomatic Triage Prior Shift (Saerens et al. Bayes adjustment) -> Selective Abstention Gate (tau=0.7852) -> Dual-Backbone Grad-CAM Saliency Projection & Bilingual Knowledge Base Lookup (webapp/service.py + webapp/knowledge_base.json) -> Interactive Result Page (webapp/static/)`.
4. **Validation-Derived Abstention Threshold (`ABSTENTION_THRESHOLD = 0.7852`):**
   - Explain how `scripts/pick_abstention_threshold.py` derived `0.7852` strictly from the 294 validation images (`runs/ensemble_seed42/val_predictions.csv`) at 85% coverage (`99.20%` Val Accuracy, `0.9940` Val Macro-F1), and include both the validation table and the in-domain test set coverage numbers (`97.07%` at 90% coverage, `99.17%` at 80% coverage).
5. **Performance, Consistency, Latency & VRAM Benchmarks:**
   - Include the 16-image consistency check results (`16/16` top-1 match, `0.0028` mean max probability difference), latency metrics (`179.4 ms` median), and GPU VRAM metrics (`1130.1 MB` single-request, `2514.3 MB` peak across test suite on RTX 3050 4 GB).
6. **Honest Model Limitations & Clinical Disclaimer:**
   - Document the 3 core limitations (Dermatitis sample size 93 train / 15 test with F1 `0.62–0.71` and merged atopic/contact etiology; SCIN external smartphone benchmark `20.77%` raw 4-class vs `44.51%` 3-class Symptomatic Triage vs `67.36%` Oracle prior bound; and what `Normal` means).
   - **Mandatory Review Flag:** Add a prominent `[!IMPORTANT]` callout flagging that the student (and ideally a medical-literate reviewer) should review the English and Thai symptom/care text in `webapp/knowledge_base.json` before recording the final video clip.
7. **Assignment References (16 Reputable Dermatology Sources + ML References):**
   - List all 16 clinical source URLs from `webapp/knowledge_base.json` (AAD, National Rosacea Society, NHS, Mayo Clinic, National Eczema Association, DermNet NZ) plus the key ML papers (He et al. ResNet, Tan & Le EfficientNet, Selvaraju et al. Grad-CAM, Guo et al. Temperature Scaling, Saerens et al. Prior Shift) so Del can paste them directly into the assignment report.

### Step 2: Write `DEMO_SCRIPT.md`
Include these sections in `DEMO_SCRIPT.md`:
1. **Pre-Recording Checklist:**
   - Starting the server (`python -m uvicorn webapp.app:app --port 8000`), opening `http://localhost:8000`, verifying `/api/health`, and a one-line PowerShell helper command that creates a tiny temporary folder `data/reports/demo_invalid_inputs/` containing `not_an_image.txt`, `corrupted_photo.jpg`, and `non_skin_blue.png` (without touching `data/raw/`) if Del wants ready-to-click files for Cases 7 and 8 during the live demo!
2. **Clip Intro (0:00 – 0:45): System Name, Logo, Privacy & Architecture:**
   - Exact talking points (EN & TH) introducing **DermaScan Face AI**, the header logo, the memory-only privacy guarantee, the medical disclaimer, and expanding the **"Model Architecture, Validation Benchmarks & Honest Known Limitations"** panel.
3. **Step-by-Step Walkthrough of All 8 Required Demo Cases:**
   For each of **Cases 1 through 8**:
   - **Goal of the Case**
   - **Exact File Path to Upload** (from `e:\CPE310FacialDermatorsisProject`) and `image_id`
   - **Expected Output on Screen** (Prediction badge, confidence %, input mode, patch count, dominant region, whether abstention or caution banner triggers, HTTP status for error cases)
   - **What to Click / Toggle on Screen** (e.g., `Grad-CAM Heatmap` / `Original Image` / `Side-by-Side`, 5-region probability table, `EN | ไทย` language switch, `Dual-Backbone Ensemble Breakdown`, `Symptomatic 3-Class Triage Mode` checkbox)
   - **Suggested Presenter Script (both English and Thai)** explaining *why* the model made that decision (linking to clinical distribution and Grad-CAM heatmaps!).

## 6. Hard Rules
- **DO NOT** touch `data/raw/`, `checkpoints/`, `.env`, `src/*`, `webapp/*`, or `tests/*`.
- Use **only in-domain `test` split images** (`split == "test"` in `data/manifest/manifest.csv`) for Demo Cases 1–6. Never use training images as proof.
- Encode both markdown files in **UTF-8**.

## 7. Acceptance Criteria
Run this verification command from `e:\CPE310FacialDermatorsisProject` and paste its exact output into your handoff report:
```powershell
$env:PYTHONIOENCODING='utf-8'; python -c "from pathlib import Path; r = Path('WEBAPP_README.md').read_text(encoding='utf-8'); d = Path('DEMO_SCRIPT.md').read_text(encoding='utf-8'); print('WEBAPP_README.md bytes:', len(r.encode('utf-8')), 'lines:', len(r.splitlines())); print('DEMO_SCRIPT.md bytes:', len(d.encode('utf-8')), 'lines:', len(d.splitlines())); assert '0.7852' in r and 'uvicorn webapp.app:app' in r and '179.4' in r, 'Missing key info in README'; paths = ['data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg', 'data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg', 'data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg', 'data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg', 'data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg', 'data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg']; [ (assert_in := (p in d), assert_exists := Path(p).exists(), print('Verified demo path:', p, 'exists:', assert_exists, 'in DEMO_SCRIPT:', assert_in)) for p in paths ]; assert all(p in d and Path(p).exists() for p in paths); print('ALL W6 DOCS & DEMO SCRIPT CHECKS PASSED')"
```
Expected: Confirms both files exist, all 6 test-split demo image paths exist on disk and are documented in `DEMO_SCRIPT.md`, and prints `ALL W6 DOCS & DEMO SCRIPT CHECKS PASSED`.

## 8. Stop Conditions
- If any of the 6 demo image paths does not exist on disk or any handoff report W1–W5 is missing, **STOP** and report that in `tasks/task_w6_handoff_report.md`.

## 9. Report Instructions
When finished, write `tasks/task_w6_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact command run and its actual output (pasted verbatim)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check in the final audit (Task W7)
- **Numbers to carry forward:** line counts and byte sizes of `WEBAPP_README.md` and `DEMO_SCRIPT.md`.

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w6_docs_and_demo_script.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** The acceptance command printed `ALL W6 DOCS & DEMO SCRIPT CHECKS PASSED` and verified all 6 demo image paths on disk.
