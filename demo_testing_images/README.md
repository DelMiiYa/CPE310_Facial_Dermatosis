# DermaScan Face AI — Demo Testing Images Directory

This folder contains all **8 benchmark test scenarios** ready for live demonstrations, presentations, and screen recordings.

> [!NOTE]
> All genuine clinical photos here are strictly from the **held-out test split** (`split == "test"` in `data/manifest/manifest.csv`). They were never seen during training or validation.

---

## 📁 File Quick Reference

| File Name | Test Target | Mode / Behavior | Expected Web Output | Key Talking Point |
|---|---|---|---|---|
| **`01_Case1_Acne_Selfie.jpg`** | Full-face Acne | 🎯 5-Patch Selfie Decomposition | **Acne (100.0%)** | MediaPipe 478 landmarks; Grad-CAM focuses on chin/cheeks (sebaceous areas). |
| **`02_Case2_Rosacea_Selfie.jpg`** | Full-face Rosacea | 🎯 5-Patch Selfie Decomposition | **Rosacea (99.99%)** | Classic "butterfly" malar flush (cheeks & nose). Differential vs adult acne. |
| **`03_Case3_Dermatitis_Macro_Caution.jpg`** | Eczema / Dermatitis | 🎯 Macro Tiling Mode | **Dermatitis (99.15%)** | **Yellow Caution Banner**: Alerts user that Dermatitis is a smaller class (93 training images). |
| **`04_Case4_Normal_Control_Face.jpeg`** | Healthy Control | 🎯 5-Patch Selfie Decomposition | **Normal Skin (100.0%)** | **Green Control Banner**: Diffuse Grad-CAM (no fake lesions); warns that normal ≠ immune to other diseases. |
| **`05_Case5_Macro_CloseUp_Tiling.jpg`** | Macro Close-up | 🎯 Automatic Tiling Mode | **Rosacea (100.0%)** | Handles close-ups without a full face; hides 5-region table cleanly. |
| **`06_Case6_Ambiguous_Lesion_Abstain.jpg`** | Ambiguous Lesion | ⚠️ Selective Abstention Gate | **Uncertain (< 78.52% threshold)** | System withholds diagnosis for safety; shows Top-2 candidates (Rosacea 40.5% vs Normal 34.1%). Toggle **Symptomatic 3-Class Triage Mode** to show Bayesian prior shift. |
| **`07_Case7_Invalid_Format_NotAnImage.txt`** | Text file upload | 🛑 Validation Layer | **HTTP 415 (`invalid_file_type`)** | Graceful bilingual error box; server stays robust without crashing. |
| **`07_Case7_Corrupted_Image.jpg`** | Truncated binary | 🛑 Validation Layer | **HTTP 400 (`unreadable_image`)** | Intercepts corrupted bytes gracefully. |
| **`08_Case8_NonSkin_SolidBlue.png`** | Solid blue image | 🛑 Skin Chrominance Filter | **HTTP 422 (`no_skin_detected`)** | YCrCb / HSV filter rejects non-skin images (prevents analyzing pets, walls, or objects). |

---

## 🚀 How to Run the Demo

1. Open your terminal in the project root:
   ```powershell
   python -m uvicorn webapp.app:app --port 8000
   ```
2. Open your web browser to:
   [`http://localhost:8000`](http://localhost:8000)
3. In the upload file dialog, navigate directly to this `demo_testing_images/` folder and pick files in numerical order (`01` through `08`).
