# Task W5 Handoff Report: Automated Tests, Consistency Check & Latency/VRAM Benchmark

- **Status:** DONE
- **Files created or changed:**
  - `tests/test_webapp.py` (created new file with 7 comprehensive automated test functions)

---

## How I Verified

Executed the exact required acceptance command from `e:\CPE310FacialDermatorsisProject`:
```powershell
$env:PYTHONIOENCODING='utf-8'; pytest tests/test_webapp.py -v -s
```

### Verbatim Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.0, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe
cachedir: .pytest_cache
rootdir: E:\CPE310FacialDermatorsisProject
plugins: anyio-4.14.2
collecting ... WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792830870.665919   18896 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792830874.274894   18896 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
collected 7 items

tests/test_webapp.py::test_health_endpoint INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1792830880.065520   24596 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1792830880.071819   15068 face_landmarker_graph.cc:180] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
W0000 00:00:1792830880.079273   18084 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1792830880.089473   26444 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
PASSED
tests/test_webapp.py::test_static_frontend_serving PASSED
tests/test_webapp.py::test_valid_selfie_and_contract_keys PASSED
tests/test_webapp.py::test_valid_closeup_and_triage_mode PASSED
tests/test_webapp.py::test_all_invalid_inputs_and_error_states PASSED
tests/test_webapp.py::test_consistency_against_test_predictions_csv 
[Consistency Check] Top-1 Agreement: 16/16 (100.00%)
[Consistency Check] Mean Max Absolute Diff: 0.0028 (Max: 0.0207)
PASSED
tests/test_webapp.py::test_eight_demo_cases_and_benchmark_summary 
==========================================================================================
                     DERMASCAN FACE AI — DEMO CASES SUMMARY
==========================================================================================
Case 1: Case 1 (Full-face selfie -> Acne)
  image_id:     iga34_65760604461e
  path:         data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg
  input_mode:   face (5 patches)
  prediction:   acne (confidence: 100.00%)
  abstained:    False
  top2:         ['acne', 'rosacea']
  latency:      1331.9 ms
------------------------------------------------------------------------------------------
Case 2: Case 2 (Full-face selfie -> Rosacea)
  image_id:     roboflow_rosacea_030fa3e083ad
  path:         data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg
  input_mode:   face (5 patches)
  prediction:   rosacea (confidence: 99.99%)
  abstained:    False
  top2:         ['rosacea', 'dermatitis']
  latency:      205.8 ms
------------------------------------------------------------------------------------------
Case 3: Case 3 (Dermatitis case)
  image_id:     roboflow_face_skin_f9ef0ad817a8
  path:         data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg
  input_mode:   closeup (6 patches)
  prediction:   dermatitis (confidence: 99.15%)
  abstained:    False
  top2:         ['dermatitis', 'rosacea']
  latency:      198.2 ms
  caution_note: Note: Dermatitis is the smallest class in the training set (93 training / 15 test images; in-domain F1 0.62–0.71) and merges atopic eczema and contact dermatitis. Interpret with extra caution.
------------------------------------------------------------------------------------------
Case 4: Case 4 (Normal face)
  image_id:     skin_assessment_fba55cb1f7a8
  path:         data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg
  input_mode:   face (5 patches)
  prediction:   normal (confidence: 100.00%)
  abstained:    False
  top2:         ['normal', 'dermatitis']
  latency:      154.0 ms
  caution_note: Note: A 'Normal' result means no strong visual signs of the 3 target conditions (Acne, Rosacea, Dermatitis) were detected. It does NOT guarantee the absence of other skin conditions.
------------------------------------------------------------------------------------------
Case 5: Case 5 (Close-up, no face detected)
  image_id:     roboflow_rosacea_c2e7137b2c96
  path:         data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg
  input_mode:   closeup (1 patches)
  prediction:   rosacea (confidence: 100.00%)
  abstained:    False
  top2:         ['rosacea', 'acne']
  latency:      125.5 ms
------------------------------------------------------------------------------------------
Case 6: Case 6 (Low-confidence -> Abstention)
  image_id:     roboflow_face_skin_4b730dec0c6d
  path:         data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg
  input_mode:   face (5 patches)
  prediction:   rosacea (confidence: 40.48%)
  abstained:    True
  top2:         ['rosacea', 'normal']
  latency:      160.6 ms
  abstention:   Uncertain — please consult a dermatologist. Top-2 candidates: Rosacea (40.5%) and Normal (34.1%).
------------------------------------------------------------------------------------------
Case 7: Invalid Input (tested 415 invalid_file_type, 400 unreadable_image, 413 file_too_large) -> VERIFIED
Case 8: Non-Skin / No-Face Input (tested 422 no_skin_detected with bilingual detail) -> VERIFIED
==========================================================================================
                     END-TO-END LATENCY & VRAM BENCHMARK
==========================================================================================
Latency Mean:   362.7 ms
Latency Median: 179.4 ms
Latency Min:    125.5 ms
Latency Max:    1331.9 ms
Peak GPU VRAM:  2514.3 MB
==========================================================================================

PASSED

============================= 7 passed in 30.64s ==============================
```

---

## Deviations from the Spec
None. Everything was built strictly in accordance with `tasks/task_w5_tests_and_benchmarks.md`.

---

## Open Issues or Doubts
None. All 7 test functions passed on the first run, the 16-image consistency check achieved 100% Top-1 agreement with mean probability difference of 0.0028 (< 0.03 spec), and peak VRAM remained well within the 4000 MB constraint (2514.3 MB peak).

---

## Numbers to Carry Forward

### 1. 16-Image Consistency Check Results
- **Top-1 Agreement:** 16 / 16 (100.00%) (spec threshold: `>= 15/16`)
- **Mean Absolute Probability Difference:** 0.0028 (spec threshold: `< 0.03`)
- **Max Absolute Probability Difference:** 0.0207

### 2. Demo Cases 1–6 Concrete Metadata (for Task W6 `DEMO_SCRIPT.md`)
| Case | Description | `image_id` | Relative File Path | `input_mode` | `num_patches` | Prediction | Confidence | `top2` | Abstained |
|---|---|---|---|---|---|---|---|---|---|
| **Case 1** | Full-face selfie -> Acne | `iga34_65760604461e` | `data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg` | `face` | 5 | `acne` | 100.00% | `['acne', 'rosacea']` | `False` |
| **Case 2** | Full-face selfie -> Rosacea | `roboflow_rosacea_030fa3e083ad` | `data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg` | `face` | 5 | `rosacea` | 99.99% | `['rosacea', 'dermatitis']` | `False` |
| **Case 3** | Dermatitis (with caution note) | `roboflow_face_skin_f9ef0ad817a8` | `data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg` | `closeup` | 6 | `dermatitis` | 99.15% | `['dermatitis', 'rosacea']` | `False` |
| **Case 4** | Normal face (with caution note) | `skin_assessment_fba55cb1f7a8` | `data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg` | `face` | 5 | `normal` | 100.00% | `['normal', 'dermatitis']` | `False` |
| **Case 5** | Close-up (no face detected, tiled patches) | `roboflow_rosacea_c2e7137b2c96` | `data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg` | `closeup` | 1 | `rosacea` | 100.00% | `['rosacea', 'acne']` | `False` |
| **Case 6** | Low-confidence -> Abstention banner | `roboflow_face_skin_4b730dec0c6d` | `data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg` | `face` | 5 | `rosacea` | 40.48% | `['rosacea', 'normal']` | `True` |

### 3. End-to-End Latency & Peak GPU VRAM
- **Latency Mean:** 362.7 ms (including initial Grad-CAM warmup; subsequent requests run in 125–205 ms)
- **Latency Median:** 179.4 ms
- **Latency Min:** 125.5 ms
- **Latency Max:** 1331.9 ms
- **Peak GPU VRAM:** 2514.3 MB (well below the 4096 MB budget)
