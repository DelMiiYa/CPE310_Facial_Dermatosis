# Handoff: Build the Demo Web App for the CPE310 Facial Dermatosis Classifier

**Audience:** The **Elephant** agent (planner/architect) working in `e:\CPE310FacialDermatorsisProject`. The Elephant does NOT write the app itself; it writes task specs that **Goldfish** agents execute one at a time. See section W (Workflow) right below, then read sections 0-10 as the requirements the specs must carry.
**Goal:** Wrap the finished model pipeline in a simple web app: the user uploads a photo, the model runs, and the page shows what the model thinks it is, how confident it is, where it looked, and educational detail about that condition and its symptoms.
**Deadline context:** University assignment (Healthcare AI System). The app will be demoed live to the professor and recorded in a clip. Reliability and clarity matter more than features.

---

## W. Workflow: Elephant-Goldfish Method

This project already used this method for Task 7 (see `tasks/TASK7_MASTER_PLAN.md`, `tasks/task_7a_efficientnet_b0.md`, `tasks/task_7a_handoff_report.md`). Follow the same file conventions and put everything in `tasks/`.

### Roles
- **Elephant** (Gemini 4 Argon, large, remembers the whole project): reads the codebase, makes design decisions, freezes interfaces, writes the master plan and one self-contained spec per Goldfish task, then reviews each Goldfish report and writes the next spec or a fix spec.
- **Goldfish** (Gemini 3.8 Flash, small, **starts with zero memory every time**): executes exactly one spec, writes a handoff report, and stops. It cannot be trusted to remember earlier tasks, infer unstated decisions, or make architecture choices.
- **Del (the human)** is the relay. Del starts each Goldfish manually in a fresh session, pastes the launch prompt, and carries the report back to the Elephant. So every spec must be copy-paste ready and every report must be readable by Del in under two minutes.

### Elephant rules
1. **Do Recon first (Task W0), yourself:** read the files listed in section 0, then write `tasks/WEBAPP_MASTER_PLAN.md` containing: the task list and dependencies, the **frozen API contract** (exact JSON field names and types returned by `/api/analyze`, and the exact keys of one `knowledge_base.json` entry), and the real function signatures you found in `src/` (what to call, with what args, returning what). Goldfish tasks depend on this, so do not guess; verify by reading the code.
2. **One spec = one deliverable.** Small enough for a Flash-class model to finish in one session (roughly 1-3 new files, under about 300 lines each). If a spec needs more, split it.
3. **Every spec is fully self-contained.** Repeat the relevant rules instead of saying "see the handoff". A Goldfish only reads its own spec plus the files the spec names.
4. **Never leave a decision open.** Name exact file paths, function names, constants, thresholds and libraries. Where Del must decide something (system name, logo), the spec says to use a placeholder constant.
5. **Freeze the contract before parallel work.** The service, API and frontend specs must all quote the same JSON contract from the master plan. The frontend must be buildable against a mock JSON file, so it never waits on the backend.
6. **Review every report before issuing the next spec.** Check the report's claims against the code (run the acceptance command yourself or ask Del for its output). If it fails, write a short fix spec instead of moving on.
7. **Keep Goldfish away from validated assets:** `data/raw/`, `checkpoints/`, existing `src/*.py` (read and import only; new code goes in new files unless the spec explicitly allows an edit), `.env`.

### Suggested task breakdown (Elephant may adjust, but keep the shape)
| ID | Task | Depends on | Output |
|---|---|---|---|
| W0 | Elephant recon and master plan, API contract frozen | none | `tasks/WEBAPP_MASTER_PLAN.md` |
| W1 | Service layer: load calibrated ensemble once, `analyze(image_bytes)` returning the contract dict (prediction, probabilities, abstention, regions, Grad-CAM as base64, latency). Includes a small script that derives the abstention threshold from the **val** split | W0 | `webapp/service.py`, `scripts/pick_abstention_threshold.py` |
| W2 | Knowledge base, EN + TH, paraphrased from reputable sources, with source URLs | W0 (schema only) | `webapp/knowledge_base.json` |
| W3 | FastAPI app: lifespan model loading, `/api/analyze`, `/api/health`, upload validation, friendly errors, static serving | W1, W2 | `webapp/app.py` |
| W4 | Frontend: upload, loading state, result view, heatmap toggle, probability bars, info card, abstention banner, disclaimer and limitations panel, mobile-friendly. Built against a mock JSON first, then pointed at the real API | W0 (contract) | `webapp/static/*` plus `webapp/static/mock_result.json` |
| W5 | Tests and measurement: API smoke tests, consistency check against `runs/ensemble_seed42/test_predictions.csv`, latency and peak VRAM | W3 | `tests/test_webapp.py`, results noted in the report |
| W6 | Docs: `WEBAPP_README.md`, `DEMO_SCRIPT.md` (the 8 cases from section 6, with real test-split image paths), and a text description of the architecture block diagram for the clip | W5 | the two docs |
| W7 | Elephant final audit: run through all 8 demo cases, check rules, write final summary for Del | W6 | `tasks/webapp_final_audit.md` |

W1 and W2 can run in parallel in two separate Goldfish sessions. W4 can start right after W0.

### Goldfish spec template (`tasks/task_w<N>_<name>.md`)
Each spec must contain these headings:
1. **Context (5 lines max):** what the project is, and where this task fits.
2. **Goal:** one sentence.
3. **Files to read first:** exact paths, plus the exact function signatures and contract snippets copied in (do not make the Goldfish hunt for them).
4. **Files to create or edit:** exact paths. Everything else is off limits.
5. **Step-by-step instructions:** numbered, concrete.
6. **Hard rules:** the relevant ones from section 8 (no touching `data/raw/`, `checkpoints/`, `.env`; UTF-8; Windows/PowerShell; no retraining; no SCIN or test data for tuning; low VRAM; load models once).
7. **Acceptance criteria:** commands Del or the Goldfish can run, with the expected output. Example: `python -c "from webapp.service import analyze; ..."` prints the keys listed in the contract.
8. **Stop conditions:** "If X is missing or a signature differs from this spec, STOP and write that in the report. Do not improvise."
9. **Report instructions:** write `tasks/task_w<N>_handoff_report.md` using the report template below, then stop.

### Goldfish handoff report template (`tasks/task_w<N>_handoff_report.md`)
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact commands run and their actual output (pasted, not summarised)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check
- **Numbers to carry forward:** thresholds, latencies, VRAM, anything later tasks need

### Launch prompt (Elephant ends each spec with a ready-to-paste block for Del)
```
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w<N>_<name>.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
Also give Del, for each task, a one-line "what to check before pasting the report back" (for example "the acceptance command printed X").

### Elephant order of operations
1. W0 now. Report to Del when `tasks/WEBAPP_MASTER_PLAN.md` is done, and list any contradictions or gaps found in this handoff.
2. Write W1, W2 and W4 specs together so Del can run them in parallel, each with its launch prompt.
3. After each report comes back, review it, update the status table in the master plan, then issue the next spec(s).
4. Finish with W7 and give Del a short final summary: what exists, how to run it, what is still uncertain.

---

## 0. Read First (do not skip)

1. `Project_handoff_summary.md` and `HANDOFF_PROMPT.md` give the full project state. The ML work is DONE. Do not retrain, re-split, or touch `data/raw/`.
2. `agent_guidelines.md` has the hard rules, including the low-VRAM constraint (RTX 3050, 4 GB).
3. Read these source files before writing any code, and reuse them. Do NOT reimplement inference:
   - `src/inference.py` (end-to-end: MediaPipe face detect -> patches -> classify -> mean-prob pooling; supports `--ensemble`)
   - `src/ensemble.py` (`DermatosisEnsemble`)
   - `src/calibration.py` (temperature scaling, Bayes prior adjustment, selective prediction)
   - `src/explainability.py` (`DermatosisExplainer`, Grad-CAM, ensemble CAM)
   - `src/detection.py`
4. Do not assume function signatures. Open the files and check how they are called, and what they return (probabilities? patch crops? face_detected flag?). If something you need is not exposed, add a thin wrapper in a NEW file (`src/webapp_service.py`). Avoid changing existing, validated modules unless unavoidable. If you must, keep the change minimal and re-run a sanity check.

---

## 1. Scope (agreed with professor)

The professor said the output should be **a detailed result from the model about what it thinks the condition is, plus detail about the symptoms and related information.** Nothing more is required.

### In scope (MVP)
- Upload one image (jpg/png/webp, size limited, e.g. 10 MB) via drag-and-drop or file picker. Mobile camera upload is a bonus (`<input type="file" accept="image/*" capture>`).
- Run the **calibrated dual-backbone ensemble** (ResNet-50 + EfficientNet-B0).
- Result page/panel showing:
  1. **Predicted condition** (Acne / Rosacea / Dermatitis / Normal) with confidence.
  2. **Probability bars for all 4 classes.**
  3. **Input handling info:** "Full-face selfie detected: 5 regions analysed (forehead, nose, left cheek, right cheek, chin)" or "Close-up image: N patches analysed".
  4. **Grad-CAM heatmap overlay** showing where the model looked (for selfies, also per-region probabilities, e.g. a small table or face-region chart).
  5. **Condition information card** for the predicted class (static knowledge base, see section 4): what it is, typical symptoms, common look/location on the face, common triggers/contributing factors, general self-care/when to see a dermatologist, and what it can be confused with (differential).
  6. **Low-confidence / abstention state:** when the model is unsure, show "Uncertain, please consult a dermatologist" with the top-2 candidates, instead of a hard answer.
  7. **Medical disclaimer**, always visible: educational demo, not a diagnosis.
- Graceful error states: no image, unsupported file, unreadable image, no skin/face found, server error.

### Out of scope (do NOT build)
- User accounts, databases, saved history, payments, deployment to cloud, mobile apps.
- Retraining or changing model weights.
- An LLM in the prediction path. See section 4 for why symptom text is static.

### Nice-to-have (only after MVP works and is tested)
- Toggle: "Symptomatic triage mode" (3-class, excludes Normal; see Step 13 in the project summary) for users who already know they have a skin problem.
- Side-by-side view: ResNet-50 vs EfficientNet-B0 vs Ensemble probabilities.
- Thai/English language toggle (the course and the professor are Thai; the info cards could have both).
- Downloadable PDF/PNG of the result.

---

## 2. Recommended Stack (keep it simple)

- **Backend:** FastAPI + Uvicorn (Python 3.13, existing venv). Single endpoint `POST /api/analyze`, plus `GET /` for the page and `GET /api/health`.
- **Frontend:** One static HTML page with vanilla JS and CSS (or a single small framework-free page). No build step, no npm. Served by FastAPI static files. Must be responsive down to phone width.
- **Alternative if time is very short:** Gradio or Streamlit. Acceptable, but a custom page looks better in the demo clip. Pick FastAPI unless there is a blocker.
- **Models loaded ONCE at startup** (FastAPI lifespan), kept on GPU if available, CPU fallback otherwise. Never reload per request. Respect the 4 GB VRAM limit: run inference under `torch.no_grad()` / autocast, process one request at a time (simple lock or single worker), and free Grad-CAM intermediates.
- New files only, suggested layout:
  ```
  webapp/
    app.py                 # FastAPI app, lifespan model loading, routes
    service.py             # wrapper around src/inference, ensemble, calibration, explainability
    knowledge_base.json    # static condition info (EN + TH)
    static/
      index.html
      app.js
      style.css
      logo.png             # project logo (see assignment requirement: system name + logo)
  tests/
    test_webapp.py         # API smoke tests
  WEBAPP_README.md         # how to run + demo script
  ```
- Add any new deps (`fastapi`, `uvicorn`, `python-multipart`) to `requirements.txt`.
- Run command target: `python -m uvicorn webapp.app:app --port 8000` from the project root, and document it.

---

## 3. Inference Behaviour (must match the validated pipeline)

Use the same settings as the Step 13 final configuration:

- Ensemble with validation-fit temperature scaling: `T_resnet = 0.60`, `T_effnet = 1.20`, `w_resnet = 0.40`. Check `src/calibration.py` (`fit_calibrated_ensemble`) for how to apply it. These values were fit on the val split only. Do not refit on test or SCIN.
- Face path: MediaPipe detects face -> 5 anatomical patches (224x224). Non-face path: tile close-up into 1-9 patches. Mean-probability pooling to image level.
- **Abstention threshold:** pick the confidence cutoff from the **validation set** (not test, not SCIN) so that roughly 80-90% of val images are answered. For reference, on the in-domain test set 80% coverage gave 99.17% accuracy and 90% coverage gave 97.07%. Store the threshold as a named constant in `service.py` with a comment on how it was derived. Report the number you chose in `WEBAPP_README.md`.
- Response JSON from `/api/analyze` should include at least:
  ```json
  {
    "prediction": "rosacea",
    "confidence": 0.91,
    "abstained": false,
    "probabilities": {"acne": 0.04, "rosacea": 0.91, "dermatitis": 0.03, "normal": 0.02},
    "top2": ["rosacea", "acne"],
    "input_mode": "face" ,
    "num_patches": 5,
    "region_probabilities": {"forehead": {...}, "nose": {...}, "left_cheek": {...}, "right_cheek": {...}, "chin": {...}},
    "gradcam_image": "data:image/png;base64,...",
    "latency_ms": 120,
    "info": { ...condition card from knowledge_base.json... },
    "disclaimer": "..."
  }
  ```
  (Adjust field names to what the existing code actually returns. `region_probabilities` only for face mode.)
- Do not save uploaded images to disk permanently (privacy). Process in memory or a temp file deleted after the request. Say so on the page.

### Known model limitations (must be shown honestly, not hidden)
- **Dermatitis is the weakest class** (only 15 images in val/test splits; F1 about 0.62-0.71 in-domain). Dermatitis predictions should carry a slightly more cautious wording in the UI.
- On the external SCIN smartphone benchmark the raw 4-class accuracy was only about 21% (mostly due to label-prior shift and close-up modality; about 44.5% with 3-class symptomatic triage). The app is a **research/educational demo**, not a validated diagnostic tool. Put a short "Limitations" section on the page or in an expandable "About this model" panel, with the real numbers from `Project_handoff_summary.md`.
- A "Normal" result means "no strong signs of the 3 target conditions detected", NOT "healthy". The model only knows 4 classes.

---

## 4. Condition Knowledge Base (static, not LLM-generated)

Create `webapp/knowledge_base.json` with one entry per class (acne, rosacea, dermatitis, normal), in English and Thai. Static text is deliberate: it is reviewable, reproducible in the demo, free, offline, and cannot hallucinate medical claims.

Each entry:
- `name`, `short_description`
- `typical_symptoms` (list)
- `typical_facial_distribution` (where on the face it usually appears; link to what Grad-CAM shows, e.g. rosacea -> central cheeks/nose, acne -> T-zone/lower face)
- `common_triggers_or_factors` (list)
- `commonly_confused_with` (list, with a one-line "how it differs", e.g. acne vs rosacea: rosacea lacks comedones and shows persistent redness and visible vessels)
- `general_care_tips` (list, conservative and non-prescriptive; no drug names or doses)
- `see_a_doctor_if` (red-flag list)
- `sources` (list of references)

Content rules:
- Base it on reputable sources (American Academy of Dermatology, National Rosacea Society, NHS, Mayo Clinic, DermNet NZ pages for text only). Do not copy text verbatim; paraphrase in your own words. Add the source URLs in `sources`. If you cannot verify a claim, leave it out.
- Dermatitis here means atopic eczema + contact dermatitis merged (as in the training labels). Say so.
- Never write "you have X". Use "The model's best match is X".
- Flag in `WEBAPP_README.md` that the Thai text and medical content should be reviewed by the student (and ideally a medical-literate person) before the final clip.
- Also reuse these sources as assignment references (the assignment needs at least 4 references).

---

## 5. UI Requirements

- Page title / header: system name + logo (the assignment's clip starts with system name and logo). If no name exists, propose one in the README (e.g. "DermaScan Face" style names), and make it a single constant so it is easy to change.
- Layout: upload panel on top or left; results below or right. After analysis show: big prediction badge + confidence, probability bars, heatmap overlay with a toggle (original / heatmap), region table (face mode), condition info card (collapsible sections), abstention banner when relevant, disclaimer footer.
- Loading state with spinner while inference runs. Disable the button during requests.
- Colour is not the only signal (accessibility). Readable fonts, sufficient contrast.
- Include 4-6 built-in "Try a sample" buttons only if you can use images that are safe to show (e.g. from the in-domain test split that are approved for use, or the project's own team photos). Do NOT embed images from restricted sources without checking licensing. If unsure, skip.

---

## 6. Demo Coverage (assignment requirement)

The clip's Demo section must "cover every case the system can handle". The app must therefore visibly handle, and the agent should prepare test images or instructions for, each of:

1. Full-face selfie -> Acne
2. Full-face selfie -> Rosacea
3. Dermatitis case (close-up or selfie)
4. Normal face
5. Close-up (no face detected) -> tiled patches path
6. Low-confidence image -> abstention banner
7. Invalid input: non-skin / non-image file / corrupted image / too large -> friendly error
8. Image with no face and no usable skin region -> friendly message

Write a `DEMO_SCRIPT.md` listing, for each case, which image path from the test split to use (pick correctly classified examples from `runs/ensemble_seed42/test_predictions.csv`, plus one abstained example), expected output, and what to say. Use in-domain TEST images for the demo; never use training images as "proof", and be honest that SCIN is the harder external set.

---

## 7. Testing

- `tests/test_webapp.py`: with FastAPI `TestClient`, check `/api/health`, a valid upload returns the expected JSON keys and probabilities that sum to about 1, an invalid file returns a 4xx with a clear message, an oversized file is rejected.
- Consistency check: for 10-20 test-split images, the web service's ensemble probabilities should match `runs/ensemble_seed42/test_predictions.csv` (allowing small numeric tolerance, remembering the saved CSVs may be pre- or post-temperature scaling; check `scripts/analyze_scin_domain_gap.py`). If they do not match, fix the integration, not the numbers.
- Measure and record end-to-end latency per request (upload to response, including Grad-CAM) and peak VRAM. Report in the README. Target is a few seconds at most on the RTX 3050.
- Manually run all 8 demo cases from section 6 and confirm each looks right.

---

## 8. Constraints and Rules

- Do not modify or delete anything in `data/raw/`, `checkpoints/`, or existing reports.
- Do not use DermNet or any clinical textbook imagery as app sample content for the demo (domain-gap rule), and do not use SCIN for tuning anything.
- `.env` holds private keys. Never print or commit them. The MVP needs no API keys.
- Keep commits small and descriptive on `main` (or a `webapp` branch): one commit for service layer, one for UI, one for knowledge base, one for tests/docs.
- Windows paths and PowerShell. Use UTF-8 for all files (Thai text!). Verify Thai renders correctly in the browser (`<meta charset="utf-8">`, a Thai-capable font such as Noto Sans Thai or Sarabun).
- Ask the user before adding anything outside the scope above.

---

## 9. Deliverables Checklist

- [ ] `webapp/` runs with one command; models load once at startup
- [ ] `/api/analyze` returns the JSON in section 3
- [ ] Calibrated ensemble + val-derived abstention threshold
- [ ] Grad-CAM overlay shown for face and close-up inputs
- [ ] `knowledge_base.json` (EN/TH) with sources, paraphrased
- [ ] Disclaimer + limitations panel with real project numbers
- [ ] Friendly error handling for all invalid inputs
- [ ] Tests passing, latency/VRAM recorded
- [ ] `WEBAPP_README.md` (setup, run, design notes, threshold value) and `DEMO_SCRIPT.md`
- [ ] Short summary back to the user of what was built, what was tested, and anything uncertain

---

## 10. Why this also helps the assignment write-up

The web app is the "Processing mode" front end. For the clip's Architecture section the flow is: **Upload -> face detection -> patch extraction -> ResNet-50 + EfficientNet-B0 -> temperature-scaled soft fusion -> confidence check (abstain or answer) -> Grad-CAM + knowledge-base card -> result page.** Keep the code structured so each step maps to one block in that diagram (the agent may also draft a block-diagram description in `WEBAPP_README.md`).
