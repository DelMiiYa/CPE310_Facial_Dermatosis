# Goldfish Task Handoff: Task 7D — SCIN Domain Gap Analysis & Post-Hoc Confidence Calibration

*Paste this file into a fresh Goldfish worker agent, or run a subagent pointed at this file. It is 100% self-contained.*

---

## 1. Goldfish Role & Strict Constraints

You are a stateless **Goldfish execution agent** working under the **Elephant & Goldfish** methodology on the CPE310 Facial Dermatosis Classification project (`e:\CPE310FacialDermatorsisProject`).
Your sole responsibility is **Task 7D (the final task of Phase 5 / Task 7): Conducting a rigorous SCIN External Test Domain Gap Analysis and Post-Hoc Confidence Calibration (`src/calibration.py` & `scripts/analyze_scin_domain_gap.py`)**, resolving the mechanistic findings from Tasks 7B and 7C.

### Hard Engineering Rules (DO NOT VIOLATE):
1. **Scope Boundary:** Execute **ONLY Task 7D**. Do **not** retrain any neural network weights. This is a fast, post-hoc statistical, calibration, and domain-gap diagnostic study using the existing checkpoints and exported prediction CSVs.
2. **Strict Zero-Leakage Calibration Rule:**
   - Any post-hoc calibration parameters intended for general model deployment — specifically **Per-Backbone Temperature Scaling ($T_{\text{resnet}}, T_{\text{effnet}}$)** and **Validation Prior / Threshold Tuning** — MUST be fit strictly on the **Validation Split (`val`, 294 images)**.
   - When analyzing the SCIN External Test Set (`external_test`, 337 images), clearly separate:
     - **(A) Blind Zero-Leakage Evaluation** (applying parameters fit only on `val` to `test` and `external_test`), and
     - **(B) Post-Hoc SCIN Domain-Shift Decomposition** (e.g., quantifying how much of the SCIN performance drop is due to **Target Label Prior Shift / Missing `Normal` Class** vs. **Capture Modality (`face_detected` selfie vs. macro close-up)** vs. **Papular/Eczematous Texture Confusion**).
3. **Untouched Raw Data, Splits & Existing Checkpoints:**
   - Never modify `data/raw/`, `data/manifest/`, `data/processed/patches/`, or existing checkpoints in `checkpoints/`.
   - Never overwrite existing reports `step9_*.md`, `step10_*.md`, `step11_*.md`, or `step12_*.md`.
4. **Hardware & Environment Guardrails (Windows, Python 3.13):**
   - Never use `cd` in PowerShell commands; run all commands with `e:\CPE310FacialDermatorsisProject` as working directory.
   - Never print or log `.env` or `ROBOFLOW_API_KEY`.

---

## 2. Context & Mechanistic Findings from Tasks 7A–7C (Entering Task 7D)

### 2.1 Existing Prediction Logs Ready on Disk (`runs/ensemble_seed42/`)
Task 7B already exported image-level probability predictions for all 3 splits:
- `runs/ensemble_seed42/val_predictions.csv` (`294` Validation images)
- `runs/ensemble_seed42/test_predictions.csv` (`303` In-Domain Test images)
- `runs/ensemble_seed42/scin_predictions.csv` (`337` SCIN External Test images, including `fst` column)

Each CSV contains:
- `image_id, subject_id, source, true_label, resnet_pred, effnet_pred, ens_equal_pred, ens_tuned_pred, models_agree`
- Per-class probabilities for all 4 classes (`Acne`, `Rosacea`, `Dermatitis`, `Normal`):
  - `resnet_prob_<Class>`
  - `effnet_prob_<Class>`
  - `ens_equal_prob_<Class>`
- You can also join `image_id` with `data/manifest/manifest.csv` to obtain `face_detected`, `face_conf`, `face_area_frac`, and `fst` for every image!

### 2.2 The Two Core Puzzles Identified in Tasks 7B & 7C
1. **Puzzle 1 — EfficientNet-B0 Overconfidence on Borderline In-Domain Disagreements:**
   - In Task 7B, across the 303 In-Domain Test images, ResNet-50 (`0.8841` Macro-F1) and EfficientNet-B0 (`0.8644` Macro-F1) disagreed on 15 images.
   - On 3 true `Dermatitis` test cases (`roboflow_face_skin_4b730dec0c6d`, `roboflow_face_skin_962a38522f33`, `roboflow_face_skin_fba72dde5494`), **ResNet-50 correctly predicted `Dermatitis`** (`47.4%–54.0%`), but **EfficientNet-B0 was overconfident on `Acne` or `Rosacea`** (`52.9%–86.2%`), pulling the uncalibrated `0.5/0.5` ensemble away from `Dermatitis`!
   - Task 7C Grad-CAM proved that EfficientNet-B0's depthwise-separable SE blocks produce pathologically sharp logit saturation on high-frequency papular edges.
   - **Hypothesis to Test:** Fitting per-backbone Temperature Scaling ($T_{\text{resnet}}, T_{\text{effnet}}$) on the `val` split (`294` images) — applying temperature $T$ to log-probabilities $\tilde{p}_c \propto p_c^{1/T}$ (i.e. $\text{softmax}(\log(p + \epsilon) / T)$) — will soften EfficientNet-B0's overconfident spikes ($T_{\text{effnet}} > T_{\text{resnet}}$) and recover In-Domain Test Macro-F1!
2. **Puzzle 2 — The SCIN External Test Dermatitis-to-Acne Domain Gap (`337` images):**
   - On the held-out SCIN benchmark (`Acne`: 58, `Rosacea`: 34, `Dermatitis`: 245, `Normal`: 0), the Ensemble achieves **`94.83%` Acne recall (55/58)** and **`0.3103` Rosacea F1**, but **220 / 245 (`89.8%`) of SCIN `Dermatitis` images are predicted as `Acne`** (and 15 as `Rosacea`, 4 as `Normal`, 6 as `Dermatitis`).
   - Why? Three distinct factors compound on SCIN:
     - **Factor A (Severe Label Prior Shift):** In our training set (`1,617` images), `Dermatitis` is only **`3.9%`** of images (`63/1,617`) while `Acne` is **`36.5%`**, `Normal` is **`30.7%`**, and `Rosacea` is **`28.8%`**. In SCIN (`337` images), `Dermatitis` is **`72.7%`** (`245/337`) and `Normal` is **`0.0%`**!
     - **Factor B (Capture Modality Shift — Selfie vs. Zoomed Macro Close-Up):** What fraction of SCIN images have `face_detected == True` vs `face_detected == False`? Without facial landmarks (`face_detected == False`), an erythematous/papular patch of contact dermatitis lacks anatomical context (e.g., perioral/eyelid distribution) and visually mimics an acne close-up tile.
     - **Factor C (Rank-2 Signal — Is `Dermatitis` or `Rosacea` the runner-up prediction?):** When SCIN `Dermatitis` is misclassified as `Acne`, what is the model's **Top-2 accuracy / recall** and what is the probability distribution of $P(\text{Dermatitis})$ on true SCIN Dermatitis cases vs. true SCIN Acne cases?

---

## 3. Step-by-Step Execution Instructions for Goldfish Agent

### Step 1: Implement `src/calibration.py`
Create [`src/calibration.py`](file:///e:/CPE310FacialDermatorsisProject/src/calibration.py) providing reusable post-hoc calibration and evaluation utilities:
1. **`compute_ece_and_brier(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict[str, Any]`:**
   - Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), multi-class Brier score, mean predictive entropy $H(p) = -\sum_c p_c \log(p_c)$, and per-bin reliability data (bin confidence, bin accuracy, bin count).
2. **`temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray`:**
   - Applies temperature scaling in log-probability space:
     $$\tilde{p}_c = \frac{\exp(\log(\max(p_c, 10^{-12})) / T)}{\sum_{j} \exp(\log(\max(p_j, 10^{-12})) / T)}$$
3. **`fit_temperature_nll(probs: np.ndarray, targets: np.ndarray) -> float`:**
   - Finds optimal scalar temperature $T^* \in [0.2, 5.0]$ minimizing negative log-likelihood (NLL) on the **Validation Split (`val`)**.
4. **`fit_calibrated_ensemble(val_res_probs: np.ndarray, val_eff_probs: np.ndarray, val_targets: np.ndarray) -> Dict[str, Any]`:**
   - Fits individual NLL-optimal temperatures $(T_{\text{resnet}}^{\text{NLL}}, T_{\text{effnet}}^{\text{NLL}})$ on `val`, AND performs a joint validation grid search over $(T_{\text{resnet}}, T_{\text{effnet}}, w_{\text{resnet}})$ strictly on `val` (`294` images) to maximize Validation Macro-F1 / minimize ECE.
5. **`apply_bayes_prior_adjustment(probs: np.ndarray, source_priors: np.ndarray, target_priors: np.ndarray) -> np.ndarray`:**
   - Implements Saerens et al. (2002) / Bayes label-shift prior correction:
     $$p_{\text{adj}}(y=c \mid x) = \frac{p(y=c \mid x) \cdot \frac{\pi_{\text{target}}(c)}{\pi_{\text{source}}(c)}}{\sum_{j} p(y=j \mid x) \cdot \frac{\pi_{\text{target}}(j)}{\pi_{\text{source}}(j)}}$$
   - Also supports **3-Class Clinical Lesion Mode (Zero-Normal Prior $\pi(\text{Normal}) = 0$)** for symptomatic dermatology triage settings (like SCIN where every patient presents with a confirmed pathological lesion).

### Step 2: Create & Run `scripts/analyze_scin_domain_gap.py`
Create [`scripts/analyze_scin_domain_gap.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/analyze_scin_domain_gap.py) that loads `runs/ensemble_seed42/{val,test,scin}_predictions.csv` and `data/manifest/manifest.csv` and executes 5 comprehensive analyses:

1. **Analysis 1: Post-Hoc Temperature Scaling & Calibration (`val` -> `test` & `scin`):**
   - Compute ECE, Brier Score, NLL, Macro-F1, and Accuracy for `ResNet-50`, `EfficientNet-B0`, `Uncalibrated Ensemble (0.5/0.5)`, and `Val-Calibrated Ensemble` across `val` (294), `test` (303), and `scin` (337).
   - Check how many of the 15 In-Domain Test disagreements (especially the 3 Dermatitis cases where EfficientNet-B0 was overconfident) are resolved after temperature/weight calibration!
2. **Analysis 2: SCIN Capture Modality Disaggregation (`face_detected == True` vs. `face_detected == False`):**
   - Join `scin_predictions.csv` with `data/manifest/manifest.csv` on `image_id`.
   - Compare SCIN performance on **Full-Face Selfies (`face_detected == True`)** vs. **Macro Close-Ups (`face_detected == False`)** across each class (`Acne`, `Rosacea`, `Dermatitis`).
3. **Analysis 3: Top-2 Diagnostic Ranking & Class Probability Margin Analysis on SCIN Dermatitis:**
   - For the 245 SCIN `Dermatitis` images:
     - Compute **Top-1 vs. Top-2 Recall** (how often is `Dermatitis` or `Rosacea` the #2 predicted class when `Acne` is #1?).
     - Compare the mean predicted `prob_Dermatitis` on true SCIN `Dermatitis` images vs. true SCIN `Acne` and `Rosacea` images (testing whether the models assign significantly higher relative `Dermatitis` probability to true Dermatitis cases even when `Acne` wins Top-1 due to prior bias!).
4. **Analysis 4: Quantifying Label Prior Shift & Symptomatic Triage Adaptation on SCIN:**
   - Compare 4 clearly labeled operating regimes on SCIN (`337` images):
     1. **Baseline Uncalibrated Ensemble** (Training prior: `Acne 36.5%, Rosacea 28.8%, Dermatitis 3.9%, Normal 30.7%`).
     2. **Zero-Leakage Validation-Calibrated Ensemble** (Temperature + threshold/prior tuned strictly on `val`).
     3. **Symptomatic Lesion Triage Mode (3-Class Uniform Disease Prior: `Normal=0%, Acne=33.3%, Rosacea=33.3%, Dermatitis=33.3%`, re-weighted against effective training priors):** Requires **zero knowledge of SCIN's label distribution**, only the clinical knowledge that the user is submitting a symptomatic lesion photo and we want equal prior sensitivity across the 3 diseases!
     4. **Oracle Target-Prior Adjusted Upper Bound (Saerens Bayes adjustment using SCIN empirical priors):** Clearly marked as an analytical upper bound to quantify how much of the SCIN gap is purely explained by label prior shift.
5. **Analysis 5: Selective Prediction / Clinical Abstention Curve (Reject Option):**
   - Evaluate accuracy and Macro-F1 on In-Domain Test (`303` images) and SCIN (`337` images) when the model abstains ("Refer to Dermatologist — Low Confidence") for images with maximum ensemble probability $<\tau$ or high predictive entropy across coverage levels ($100\%, 90\%, 80\%, 70\%$).

### Step 3: Generate Diagnostic Plots & Final Report (`data/reports/step13_scin_domain_gap_report.md`)
1. Save publication-quality diagnostic plots to `data/reports/calibration_plots/`:
   - `reliability_diagram_test_and_scin.png` (Reliability curves & confidence histograms before/after temperature scaling for ResNet-50, EfficientNet-B0, and Ensemble).
   - `scin_domain_gap_breakdown.png` (Bar/distribution plots showing: (a) SCIN Face-Detected vs Close-Up accuracy, (b) Top-1 vs Top-2 recall on SCIN, (c) Prior-adjustment progression from Uncalibrated -> Val-Calibrated -> Uniform-3-Class Disease Prior -> Oracle Target Prior, and (d) Selective Abstention accuracy-vs-coverage curve).
2. Save the comprehensive markdown report to [`data/reports/step13_scin_domain_gap_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step13_scin_domain_gap_report.md) with:
   - **Section 1:** Executive Summary of the SCIN Domain Gap & Calibration Study.
   - **Section 2:** Post-Hoc Temperature Scaling & Calibration Results (`val` & `test` & `scin` ECE, Brier Score, NLL, and In-Domain Test Macro-F1 / Disagreement Resolution).
   - **Section 3:** Mechanistic Decomposition of the SCIN External Test Gap (Capture Modality `face_detected` vs. Close-Up, Top-1 vs. Top-2 Recall, and Relative Probability Signal).
   - **Section 4:** Label Prior Shift & Symptomatic Triage Adaptation Table (comparing Uncalibrated vs. Val-Calibrated vs. Zero-Leakage 3-Class Uniform Disease Prior vs. Oracle SCIN Prior across Macro-F1, Accuracy, and Dermatitis Recall/F1).
   - **Section 5:** Clinical Selective Abstention (Coverage vs. Accuracy / Macro-F1 Trade-off Table).
   - **Section 6:** Final Project Conclusions & Clinical Deployment Recommendations (including justification for merging Atopic Eczema + Contact Dermatitis into `Dermatitis` per the literature survey).

---

## 4. Acceptance Criteria & Handoff Back to Elephant

Before finishing, verify all items below:
- [ ] [`src/calibration.py`](file:///e:/CPE310FacialDermatorsisProject/src/calibration.py) and [`scripts/analyze_scin_domain_gap.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/analyze_scin_domain_gap.py) are implemented and run cleanly in `< 30 seconds`.
- [ ] Diagnostic figures exist in `data/reports/calibration_plots/`.
- [ ] Comprehensive report exists at [`data/reports/step13_scin_domain_gap_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step13_scin_domain_gap_report.md) containing all 6 required sections.
- [ ] Existing checkpoints, manifests, and raw data (`data/raw/`) are completely untouched.
- [ ] Save a handoff summary to `tasks/task_7d_handoff_report.md` and return a concise summary of findings to the Elephant coordinator.
