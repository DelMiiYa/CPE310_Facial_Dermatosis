# CPE310 Facial Dermatosis Classifier

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-008080.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interpretable, deep-learning classification pipeline engineered specifically to diagnose facial skin conditions from **unconstrained smartphone imagery** (both full-face selfies and macro close-ups).

Developed for **CPE310** (King Mongkut's University of Technology Thonburi), this project addresses the severe **domain gap** observed between textbook clinical dermoscopy and real-world consumer smartphone cameras.

---

## 🌟 Key Features

* **Real-World Domain Focus:** Strictly trained on smartphone-captured imagery to avoid the clinical dermoscopy domain gap (which previously collapsed to predicting "Normal" on real user selfies).
* **Hybrid Patch-Based Pipeline:** 
  * **Full-Face Selfies:** MediaPipe Face Landmarker (478 landmarks) extracts 5 standardized anatomical regions: **Forehead, Nose, Left Cheek, Right Cheek, Chin** ($224 \times 224$).
  * **Macro Close-Ups:** Automatic tiling into standardized $224 \times 224$ patches.
  * **Image-Level Diagnosis:** Mean-probability pooling across all extracted patches.
* **Dual-Backbone Soft Voting Ensemble:** Fuses **ResNet-50** and **EfficientNet-B0** ($w^* = 0.50 / 0.50$), achieving **92.41% Test Accuracy** and resolving 33.3% of single-model disagreements with **16.39 ms** latency.
* **Clinical Interpretability (Grad-CAM):** Quantitative spatial attribution across anatomical regions verifies clinical alignment (e.g., 68.1% of Rosacea attributions localize to the malar/flush zone: nose and cheeks).
* **Post-Hoc Confidence Calibration & Domain Gap Deconstruction:** Zero-leakage temperature scaling cuts Expected Calibration Error (ECE) by **36%** (to `0.0197`). Evaluated against the Google SCIN external benchmark with Symptomatic 3-Class Triage Mode and selective abstention (reaching **99.17% accuracy** at 80% patient coverage).
* **Low-VRAM Optimization:** Fully runnable on **4 GB VRAM** GPUs (NVIDIA RTX 3050 Laptop) via mixed-precision AMP (`torch.amp.autocast`) and optimized PyTorch DataLoaders.

---

## 🩺 Target Conditions (4-Class Problem)

| Class | Clinical Condition | Description |
|---|---|---|
| **Acne** | *Acne Vulgaris* | Inflammatory papules, pustules, and comedones across the sebaceous T-zone. |
| **Rosacea** | Erythematotelangiectatic / Papulopustular | Persistent centrofacial erythema, telangiectasia, and inflammatory lesions over the cheeks and nose. |
| **Dermatitis** | Atopic Eczema + Contact Dermatitis | Merged per dermatological literature; pruritic erythematous patches with scaling and micro-papules. |
| **Normal** | Healthy Facial Skin | Control class of healthy facial skin. |

---

## 📊 Benchmark Results

### 1. In-Domain Test Set (303 Images / 222 Subjects)

Grouped strictly by `subject_id` with zero patient overlap across 70/15/15 train/val/test splits:

| Architecture | Parameters | Checkpoint Size | Accuracy | Macro-F1 | Acne F1 | Rosacea F1 | Dermatitis F1 | Normal F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-50** | 23.52M | 269.6 MB | **92.74%** | **0.8841** | 0.9412 | 0.9091 | **0.7059** | 0.9804 |
| **EfficientNet-B0** | 4.01M | 46.4 MB | **92.74%** | 0.8644 | **0.9464** | **0.9100** | 0.6207 | 0.9804 |
| **Ensemble ($w^*=0.5/0.5$)** | 27.53M | 316.0 MB | 92.41% | 0.8621 | 0.9417 | 0.9055 | 0.6207 | 0.9804 |
| **Calibrated Ensemble ($T_{\text{res}}=0.6, T_{\text{eff}}=1.2$)** | 27.53M | 316.0 MB | **92.74%** | 0.8736 | 0.9457 | 0.9045 | 0.6452 | 0.9804 |

* **Selective Prediction (Abstention Option):** Withholding ambiguous predictions for dermatologist review yields **97.07% accuracy** at 90% coverage and **99.17% accuracy (240/242 correct)** at 80% coverage.

### 2. SCIN Held-Out External Test Set (337 Images / 149 Cases)

The crowd-sourced Google SCIN dataset was held out strictly as an unseen real-world external benchmark:

* **Symptomatic 3-Class Triage Mode:** Doubled SCIN accuracy from 20.77% to **44.51%** and surged Dermatitis recall from 2.45% to **45.31%** (reaching **86.12%** under the target prior bound).
* **Skin Tone Fairness:** Verified across Fitzpatrick Skin Types (`FST1` to `FST5`), with highest accuracy on darker skin tones (`FST4`: 31.6%, `FST5`: 36.0%), confirming robust generalization.

---

## 🔍 Visual Interpretability (Grad-CAM)

Quantitative anatomical region attribution across 224 test selfies:
* **Rosacea:** Centrofacial flush focus — **68.1%** of dominant activations occur in the malar/nose regions (`nose`: 22.7%, `left_cheek`: 22.7%, `right_cheek`: 22.7%).
* **Acne:** Concentrated on sebaceous follicles across the chin (28.9%) and lateral cheeks (40.2%).
* **Normal:** Uniformly dispersed probabilities (96.7%–97.9%) with zero focal lesion artifacts.

Interactive visual review galleries are located in `data/reports/galleries/step12_gradcam_gallery.html`.

---

## 🏗️ Repository Architecture

```
CPE310_Facial_Dermatosis/
├── checkpoints/              # Pretrained model checkpoints (git-ignored)
│   ├── resnet50_baseline_best.pt
│   └── efficientnet_b0_baseline_best.pt
├── data/                     # Raw and processed datasets (git-ignored)
│   ├── manifest/             # manifest.csv, patches_manifest.csv
│   ├── processed/patches/    # Pre-extracted 224x224 patches (train, val, test, scin)
│   └── reports/              # Detailed evaluation markdown reports & Grad-CAM plots
├── scripts/                  # Standalone execution pipelines
│   ├── run_face_detection.py # MediaPipe face landmark detection
│   ├── extract_patches.py    # 224x224 anatomical patch cropper
│   ├── deduplicate.py        # 64-bit pHash deduplication & patient clustering
│   ├── split_dataset.py      # Grouped-by-subject 70/15/15 stratified splitter
│   ├── train_baseline.py     # Parameterized training engine (ResNet-50 / EfficientNet-B0)
│   ├── evaluate_ensemble.py  # Dual-backbone soft voting & weight tuning
│   ├── generate_gradcam.py   # Multi-panel Grad-CAM visualization generator
│   └── analyze_scin_domain_gap.py # Calibration, prior shift, and abstention analysis
├── src/                      # Core modular library
│   ├── detection.py          # MediaPipe anatomical cropping logic
│   ├── dataset.py            # PyTorch patch & image datasets with fast memory indexing
│   ├── model.py              # timm wrappers & patch-pooling classifier
│   ├── ensemble.py           # Dual-backbone soft probability fusion
│   ├── explainability.py     # ResNet, EfficientNet, and Ensemble Grad-CAM explainer
│   ├── calibration.py        # Temperature scaling, ECE/Brier scores, and Bayes priors
│   └── inference.py          # End-to-end inference on unconstrained user photos
├── tasks/                    # Engineering specifications & handoff logs
├── Project_handoff_summary.md# Comprehensive technical summary across all steps
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.13)
* NVIDIA GPU with CUDA support recommended (e.g., RTX 3050 Laptop 4GB)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/DelMiiYa/CPE310_Facial_Dermatosis.git
cd CPE310_Facial_Dermatosis

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage & Inference

### 1. Single Image Inference
Run inference directly on any raw smartphone selfie or close-up macro photo:

```powershell
# Standard ResNet-50 baseline inference
python src/inference.py --image "path/to/selfie.jpg"

# Lightweight EfficientNet-B0 inference
python src/inference.py --image "path/to/selfie.jpg" --checkpoint checkpoints/efficientnet_b0_baseline_best.pt

# Dual-backbone soft voting ensemble inference
python src/inference.py --image "path/to/selfie.jpg" --ensemble
```

Sample output:
```json
{
  "predicted_class": "Acne",
  "confidence": 0.9421,
  "probabilities": {
    "Acne": 0.9421,
    "Rosacea": 0.0418,
    "Dermatitis": 0.0135,
    "Normal": 0.0026
  },
  "face_detected": true,
  "num_patches": 5
}
```

### 2. Grad-CAM Heatmap Generation
Generate visual saliency maps highlighting the lesion features driving the prediction:

```powershell
python src/explainability.py --image "path/to/selfie.jpg" --output "gradcam_output.png"
```

### 3. Evaluate Confidence Calibration & Domain Gap
Reproduce the Expected Calibration Error (ECE) curves, symptomatic triage mode, and clinical selective abstention analysis:

```powershell
python scripts/analyze_scin_domain_gap.py
```

---

## 📖 Citation & References

* **MediaPipe Tasks Vision:** Face Landmarker & BlazeFace for anatomical skin cropping.
* **timm:** PyTorch Image Models (ResNet-50, EfficientNet-B0).
* **Google SCIN Dataset:** Crowd-sourced smartphone dermatological dataset used as external benchmark.
* **CPE310 Facial Dermatosis Project:** Department of Computer Engineering, KMUTT.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
