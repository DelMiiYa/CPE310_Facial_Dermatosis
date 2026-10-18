# Elephant Master Plan: Task 7 (Advanced Architectures, Ensemble & Explainability)

**Architecture Methodology:** Elephant & Goldfish  
- **Elephant (Coordinator / Planner):** Maintains global project context, verifies deliverables against [`Project_handoff_summary.md`](file:///e:/CPE310FacialDermatorsisProject/Project_handoff_summary.md) and [`HANDOFF_PROMPT.md`](file:///e:/CPE310FacialDermatorsisProject/HANDOFF_PROMPT.md), writes self-contained Goldfish task handoff files, and updates project state after each milestone.
- **Goldfish (Worker Agent):** Stateless execution agent given a single, self-contained task specification file (`tasks/task_7x_*.md`). Executes only that atomic unit of work, produces a verification report, commits on approval, and terminates.

---

## Current Project Baseline (Entering Task 7)

- **Completed Phases:** Steps 1–9 (Data ingestion, MediaPipe face detection & 14,330 patch extraction, pHash deduplication, manual review of 2,214 training images, 70/15/15 subject-grouped stratified split, and ResNet-50 baseline training).
- **ResNet-50 Baseline (`checkpoints/resnet50_baseline_best.pt`, Seed 42, Epoch 11):**
  - **In-Domain Test (303 images / 222 subjects):** Macro-F1 = **`0.8841`**, Accuracy = **`92.74%`** (Acne F1: `0.9412`, Rosacea F1: `0.9091`, Dermatitis F1: `0.7059`, Normal F1: `0.9804`).
  - **SCIN Held-Out External Test (337 images / 149 cases):** Macro-F1 = `0.1544`, Accuracy = `21.66%` (Acne Recall: `91.38%`, Rosacea Recall: `20.59%`, Dermatitis Recall: `5.31%`).

---

## Task 7 Decomposition into Goldfish Sub-Tasks

```mermaid
flowchart LR
    T7A["Task 7A: EfficientNet-B0\nTraining & Benchmark"] --> T7B["Task 7B: ResNet-50 + EfficientNet-B0\nEnsemble & Multi-Seed"]
    T7B --> T7C["Task 7C: Grad-CAM\nVisual Explainability"]
    T7C --> T7D["Task 7D: SCIN Domain Gap\n& Calibration Analysis"]
```

### Goldfish Task 7A: EfficientNet-B0 Training & Head-to-Head Comparison (COMPLETED)
- **Handoff Spec:** [`tasks/task_7a_efficientnet_b0.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7a_efficientnet_b0.md)
- **Handoff Report:** [`tasks/task_7a_handoff_report.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7a_handoff_report.md)
- **Status:** **COMPLETED** (Seed 42, Best Epoch 9, Val Macro-F1: 0.9227)
- **Key Deliverables Delivered:**
  - `scripts/train_baseline.py` parameterized with `--backbone`; `scripts/train_efficientnet.py` created.
  - `checkpoints/efficientnet_b0_baseline_best.pt` and `checkpoints/efficientnet_b0_seed42_best.pt` saved.
  - `runs/efficientnet_b0_seed42/training_log.csv` recorded for all 15 epochs.
  - Head-to-head evaluation report at `data/reports/step10_efficientnet_b0_report.md`.
  - In-Domain Test: **92.74% Accuracy**, **0.8644 Macro-F1** (Identical accuracy to ResNet-50 with 82.9% fewer parameters).
  - Theoretical Ensemble Oracle Accuracy: **94.72%** (+1.98% headroom over single models).

### Goldfish Task 7B: ResNet-50 + EfficientNet-B0 Model Ensemble (COMPLETED)
- **Handoff Spec:** [`tasks/task_7b_model_ensemble.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7b_model_ensemble.md)
- **Report:** [`data/reports/step11_ensemble_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step11_ensemble_report.md)
- **Status:** **COMPLETED** (Optimal Validation Weights $w^* = 0.50 / 0.50$, Val Macro-F1: **0.9264**, Val Acc: **95.92%**)
- **Key Deliverables Delivered:**
  - [`src/ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/src/ensemble.py), [`scripts/evaluate_ensemble.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/evaluate_ensemble.py), and `--ensemble` flag in [`src/inference.py`](file:///e:/CPE310FacialDermatorsisProject/src/inference.py).
  - Saved per-image prediction CSVs in `runs/ensemble_seed42/{val,test,scin}_predictions.csv`.
  - In-Domain Test: **92.41% Accuracy (280/303)**, **0.8621 Macro-F1**, resolving **5/15 (33.3%)** single-model disagreements at **16.39 ms** 5-patch latency.
  - SCIN External Test: Achieved best-in-project **0.1665 Macro-F1** (**0.3103 Rosacea F1**, **94.83% Acne recall**).

### Goldfish Task 7C: Grad-CAM Visual Interpretability (COMPLETED)
- **Handoff Spec:** [`tasks/task_7c_gradcam_explainability.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7c_gradcam_explainability.md)
- **Handoff Report:** [`tasks/task_7c_handoff_report.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7c_handoff_report.md)
- **Report & Gallery:** [`data/reports/step12_gradcam_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step12_gradcam_report.md) | [`data/reports/galleries/step12_gradcam_gallery.html`](file:///e:/CPE310FacialDermatorsisProject/data/reports/galleries/step12_gradcam_gallery.html)
- **Status:** **COMPLETED** (16 multi-panel diagnostic figures generated across Cohorts A–D + quantitative attribution across 224 test selfies)
- **Key Deliverables Delivered:**
  - [`src/explainability.py`](file:///e:/CPE310FacialDermatorsisProject/src/explainability.py) and [`scripts/generate_gradcam.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/generate_gradcam.py) implemented.
  - Quantitative attribution confirmed centrofacial/malar dominance (`nose`, `left_cheek`, `right_cheek` = `68.1%`) in Rosacea and T-zone/cheek dominance in Acne.
  - Mechanistically proved that EfficientNet-B0's depthwise-separable SE blocks over-activate on circular micro-papular edges in Dermatitis, motivating per-backbone Temperature Scaling in Task 7D.

### Goldfish Task 7D: SCIN Domain Gap Analysis & Confidence Calibration (COMPLETED)
- **Handoff Spec:** [`tasks/task_7d_scin_domain_analysis.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7d_scin_domain_analysis.md)
- **Handoff Report:** [`tasks/task_7d_handoff_report.md`](file:///e:/CPE310FacialDermatorsisProject/tasks/task_7d_handoff_report.md)
- **Report & Plots:** [`data/reports/step13_scin_domain_gap_report.md`](file:///e:/CPE310FacialDermatorsisProject/data/reports/step13_scin_domain_gap_report.md) | [`data/reports/calibration_plots/`](file:///e:/CPE310FacialDermatorsisProject/data/reports/calibration_plots/)
- **Status:** **COMPLETED** (Executed in 3.87s under strict zero-leakage validation fitting)
- **Key Deliverables Delivered:**
  - [`src/calibration.py`](file:///e:/CPE310FacialDermatorsisProject/src/calibration.py) and [`scripts/analyze_scin_domain_gap.py`](file:///e:/CPE310FacialDermatorsisProject/scripts/analyze_scin_domain_gap.py) implemented.
  - Validation-tuned temperature scaling ($T_{\text{resnet}}=0.60, T_{\text{effnet}}=1.20, w_{\text{resnet}}=0.40$) increased Val Macro-F1 to **0.9372**, lifted In-Domain Test Macro-F1 to **0.8736** (**92.74% Acc**), and cut In-Domain Test ECE by **36.0% (0.0197)**.
  - Zero-leakage Symptomatic 3-Class Triage Mode ($\pi(\text{Normal})=0$, equal $1/3$ disease priors) more than doubled SCIN External Test Accuracy (**20.77% -> 44.51%**) and Macro-F1 (**0.1665 -> 0.3321**), surging Dermatitis recall from **2.45% to 45.31%** (**86.12%** under Oracle target prior).
  - Selective abstention at 80% coverage achieved **99.17% In-Domain Test Accuracy (240/242 correct, 0.9464 Macro-F1)**.
