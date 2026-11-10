# DermaScan Face AI — Comprehensive Live Demo & Video Presentation Script

[![CPE310 Healthcare AI](https://img.shields.io/badge/CPE310-Healthcare%20AI%20Project-blue.svg)](#)
[![Live Demo Script](https://img.shields.io/badge/Demo%20Cases-8%20Scenarios-brightgreen.svg)](#)
[![Bilingual](https://img.shields.io/badge/Languages-EN%20%7C%20TH-teal.svg)](#)
[![Validation Threshold](https://img.shields.io/badge/%CF%84-0.7852-orange.svg)](#)

---

## 1. Pre-Recording Checklist & Live Environment Setup

Before starting the video recording or live demonstration, verify each item on this checklist:

### 1.1 Server & Hardware Verification
1. **Operating Environment:** Windows 11 with NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), Python 3.13, and PyTorch 2.6 (`+cu124`).
2. **Start the FastAPI Application Server (Terminal 1):**
   ```powershell
   python -m uvicorn webapp.app:app --port 8000
   ```
   *Expected output:* Confirms Uvicorn running on `http://127.0.0.1:8000` with lifespan model loading complete.
3. **Verify System Health Endpoint:**
   Open a browser or new PowerShell window and run:
   ```powershell
   curl http://localhost:8000/api/health
   ```
   *Expected response:* `{"status": "ok", "system_name": "DermaScan Face AI", "device": "cuda", "models_loaded": true, "knowledge_base_loaded": true, "abstention_threshold": 0.7852, ...}`.
4. **Open Application in Web Browser:**
   Navigate to [`http://localhost:8000`](http://localhost:8000). Set zoom level to 100% or 110% for optimal video readability.

### 1.2 One-Line Helper: Generate Demo Test Files for Cases 7 & 8
To have convenient, ready-to-click invalid input files for Cases 7 and 8 without modifying anything in `data/raw/`, run this one-line command from the project root:

```powershell
python -c "from pathlib import Path; import cv2, numpy as np; d = Path('data/reports/demo_invalid_inputs'); d.mkdir(parents=True, exist_ok=True); (d / 'not_an_image.txt').write_text('This is a plain text file, not a facial image.', encoding='utf-8'); (d / 'corrupted_photo.jpg').write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00corrupted_bytes_here'); cv2.imwrite(str(d / 'non_skin_blue.png'), np.full((300, 300, 3), (255, 0, 0), dtype=np.uint8)); print('Created demo invalid files in', d)"
```

This creates three tiny test files in `data/reports/demo_invalid_inputs/`:
* `not_an_image.txt` (Triggers HTTP 415 `invalid_file_type`)
* `corrupted_photo.jpg` (Triggers HTTP 400 `unreadable_image`)
* `non_skin_blue.png` (Triggers HTTP 422 `no_skin_detected`)

---

## 2. In-Domain Test Split Integrity Notice

> [!IMPORTANT]
> **Scientific Integrity & Zero Training Contamination:**
> All test images specified for **Demo Cases 1 through 6** below belong strictly to the **held-out in-domain `test` split** (`split == "test"` in `data/manifest/manifest.csv`).
> 
> *Note on folder naming:* Certain raw files reside in folders named `.../train/...` on disk solely because Roboflow's original download zip archive organized its files into subfolders with those names. Our data engineering pipeline completely discarded Roboflow's partition, deduplicated all images using 64-bit perceptual hashing (`pHash`), and generated a clean, subject-grouped 70/15/15 stratified split in `manifest.csv`. Every image used in this demo has never been seen by either backbone during training or validation.

---

## 3. Video Clip Introduction (0:00 – 0:45)

### Screen Setup:
* Browser showing `http://localhost:8000` with the clean header, brand logo, and the always-visible Medical & Privacy notice banner.
* Expand the bottom collapsible panel: **"Model Architecture, Validation Benchmarks & Honest Known Limitations"**.

### Presenter Talking Points:

```text
[English]
"Hello everyone, welcome to the demonstration of DermaScan Face AI, developed for our CPE310 Healthcare AI project.

DermaScan Face AI is a clinical decision-support web application designed to analyze real-world smartphone photos for four target facial skin conditions: Acne Vulgaris, Rosacea, Dermatitis, and Normal Skin.

Notice three foundational design choices on our interface:
1. Strict Privacy Guarantee: All photos are processed in volatile RAM and immediately discarded. No user photos are ever saved to disk.
2. Clinical Safety & Medical Governance: The app is an educational decision-support tool, not a standalone diagnostic device.
3. Dual-Backbone Architecture & Calibration: As seen in our architecture panel, the backend fuses a 23.5-million-parameter ResNet-50 and a 4-million-parameter EfficientNet-B0. Both models undergo post-hoc temperature scaling fitted strictly on 294 held-out validation images, and are paired with an aligned Grad-CAM visual explainability engine.

Let's now walk through our 8 live evaluation cases."
```

```text
[ไทย]
"สวัสดีครับทุกท่าน ขอต้อนรับสู่การนำเสนอระบบ DermaScan Face AI สำหรับโครงงานวิชา CPE310 Healthcare AI System ครับ

DermaScan Face AI คือระบบเว็บแอปพลิเคชันสนับสนุนการตัดสินใจทางคลินิก ที่ถูกออกแบบมาเพื่อวิเคราะห์ภาพถ่ายจากสมาร์ตโฟนจริง ทั้งภาพเซลฟี่เต็มหน้าและภาพถ่ายรอยโรคระยะใกล้ สำหรับ 4 ภาวะผิวหน้า ได้แก่ สิว (Acne), โรคโรซาเซีย (Rosacea), โรคผิวหนังอักเสบ (Dermatitis) และผิวปกติ (Normal Skin)

บนหน้าจอหลักของเราจะเห็นจุดเด่นสำคัญ 3 ประการ:
1. การคุ้มครองความเป็นส่วนตัวอย่างเข้มงวด: ภาพถ่ายทั้งหมดจะถูกประมวลผลบนหน่วยความจำชั่วคราว (RAM) และลบทิ้งทันที โดยไม่มีการบันทึกภาพลงฮาร์ดดิสก์
2. ธรรมาภิบาลและความปลอดภัยทางการแพทย์: ระบบมีแถบเตือนชัดเจนว่าเป็นเครื่องมือช่วยตัดสินใจเพื่อการศึกษา ไม่ใช่อุปกรณ์วินิจฉัยโรคขั้นสุดท้าย
3. สถาปัตยกรรมโมเดลคู่และการปรับเทียบความเชื่อมั่น: ในแถบข้อมูลโมเดลด้านล่าง เราผสานสองโครงข่ายประสาทเทียมชั้นนำคือ ResNet-50 ขนาด 23.5 ล้านพารามิเตอร์ และ EfficientNet-B0 ขนาด 4 ล้านพารามิเตอร์ พร้อมการปรับเทียบอุณหภูมิ (Temperature Scaling) จากชุด Validation 294 ภาพ และมีระบบ Grad-CAM อธิบายผลลัพธ์ด้วยภาพ

ต่อไป เรามาชมการทดสอบทั้ง 8 กรณีศึกษาไปพร้อมกันครับ"
```

---

## 4. Step-by-Step Walkthrough: All 8 Required Demo Cases

---

### Case 1: Full-Face Selfie $\rightarrow$ Acne Vulgaris (Sebaceous Distribution)

* **Goal:** Demonstrate full-face detection, 478-landmark extraction, 5 anatomical region decomposition, and high-confidence Acne detection with aligned Grad-CAM saliency on active inflammatory lesions.
* **Exact File Path:**  
  `data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg`
* **Image ID:** `iga34_65760604461e` (`split == "test"`)
* **Expected Output on Screen:**
  * **Prediction Badge:** `Acne` (Teal badge)
  * **Calibrated Confidence:** `100.00%`
  * **Input Mode Badge:** `🎯 Full-face selfie detected: 5 anatomical regions analysed`
  * **Abstention Status:** `abstained = False` (No abstention banner)
  * **Top-2 Candidates:** `Acne` (100.0%) and `Rosacea` (<0.1%)
  * **Latency:** ~150–205 ms steady-state
  * **Dominant Region:** Chin / Cheek regions
* **What to Click / Toggle on Screen:**
  1. Click **"Analyze Photo"** and observe the ~180 ms response time.
  2. Toggle the segmented viewer between **"Grad-CAM Heatmap"**, **"Original Image"**, and **"Side-by-Side"**.
  3. Inspect the **"Anatomical Region Probability Breakdown"** table, showing per-zone probabilities across forehead, nose, left cheek, right cheek, and chin.
  4. Toggle the language button (**"🌐 ไทย (TH)"** $\leftrightarrow$ **"🌐 English"**) to demonstrate instant bilingual translation of symptoms, triggers, care tips, and doctor visit recommendations.
  5. Expand the **"Dual-Backbone Ensemble Breakdown"** to show both ResNet-50 and EfficientNet-B0 agreeing on Acne.

#### Suggested Presenter Script:

```text
[English]
"In Case 1, we upload a genuine smartphone selfie from a patient with acne. 

MediaPipe Tasks Vision instantly identifies the face and maps 478 landmarks, decomposing the face into five standardized 224x224 anatomical regions: forehead, nose, left cheek, right cheek, and chin. Both ResNet-50 and EfficientNet-B0 agree 100%, predicting Acne Vulgaris with 100.00% calibrated confidence.

Now look at the Grad-CAM saliency map. Notice how the activation focuses tightly on the active inflammatory papules and comedones on the chin and lateral cheeks, while remaining completely silent over the background, eyes, and hair. 

In the anatomical breakdown table, the chin and cheeks show the highest acne probabilities. This directly mirrors clinical dermatology findings: acne vulgaris concentrates in high-sebaceous areas. Finally, switching to Thai demonstrates that our knowledge base and care tips translate seamlessly."
```

```text
[ไทย]
"ในเคสแรก เราอัปโหลดภาพถ่ายเซลฟี่จริงจากสมาร์ตโฟนของผู้ที่มีปัญหาสิวครับ

ระบบ MediaPipe ตรวจพบใบหน้าและประมวลผล 478 จุดแลนด์มาร์ก เพื่อแยกแผ่นภาพกายวิภาคศาสตร์ 5 โซนขนาด 224x224 พิกเซล ได้แก่ หน้าผาก จมูก แก้มซ้าย แก้มขวา และคาง โมเดลทั้งสองตัวทำนายตรงกันว่าเป็น สิว (Acne Vulgaris) ด้วยความมั่นใจที่ผ่านการปรับเทียบ 100.00%

เมื่อดูภาพ Grad-CAM แบบเคียงข้างกัน (Side-by-Side) จะเห็นชัดเจนว่าจุดความร้อนโฟกัสอย่างแม่นยำที่ตุ่มสิวอักเสบและสิวอุดตันบริเวณคางและแก้ม โดยไม่สนใจพื้นหลัง ดวงตา หรือเส้นผมเลยครับ

ตารางแยกตามโซนกายวิภาคยังแสดงให้เห็นว่าบริเวณคางและแก้มมีค่าความน่าจะเป็นสูงสุด ซึ่งตรงกับลักษณะการกระจายตัวของต่อมไขมันในผู้ป่วยสิว และเมื่อกดปุ่มเปลี่ยนภาษา ข้อมูลอาการ ปัจจัยกระตุ้น และแนวทางการดูแลตัวเองจะแสดงผลเป็นภาษาไทยอย่างสมบูรณ์ครับ"
```

---

### Case 2: Full-Face Selfie $\rightarrow$ Rosacea (Centrofacial Malar Attribution)

* **Goal:** Demonstrate centrofacial erythema detection in Rosacea and show that the Grad-CAM attention aligns with the clinical butterfly / malar distribution.
* **Exact File Path:**  
  `data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg`
* **Image ID:** `roboflow_rosacea_030fa3e083ad` (`split == "test"`)
* **Expected Output on Screen:**
  * **Prediction Badge:** `Rosacea` (Rose-pink badge)
  * **Calibrated Confidence:** `99.99%`
  * **Input Mode Badge:** `🎯 Full-face selfie detected: 5 anatomical regions analysed`
  * **Abstention Status:** `abstained = False`
  * **Top-2 Candidates:** `Rosacea` (99.99%) and `Dermatitis` (0.01%)
  * **Dominant Region:** Centrofacial (`nose`, `left_cheek`, or `right_cheek`)
* **What to Click / Toggle on Screen:**
  1. Click **"Side-by-Side"** view in the explainability card to contrast the raw facial redness with the Grad-CAM heatmap.
  2. Point out the **"Differential Diagnosis"** block in the knowledge base, highlighting how Rosacea differs from Acne (absence of comedones; presence of facial flushing, telangiectasia, and ocular involvement).
  3. Show the **Reputable References** section with direct clickable links to the *National Rosacea Society* and *American Academy of Dermatology*.

#### Suggested Presenter Script:

```text
[English]
"In Case 2, we upload a full-face selfie presenting with widespread facial erythema. 

DermaScan Face AI diagnoses Rosacea with 99.99% confidence. When we look at the Grad-CAM heatmap in Side-by-Side view, the activation forms a classic butterfly pattern centered across the nasal bridge and medial cheeks. 

In clinical practice, Rosacea is frequently confused with Adult Acne. However, our model identifies the key differential: persistent erythema and telangiectasia without the comedones characteristic of acne. In our Step 12 quantitative evaluation across 224 test selfies, 68.1% of dominant patch attributions for Rosacea concentrated in this centrofacial zone. The visual heatmap confirms that the neural network has learned genuine clinical markers rather than spurious background artifacts."
```

```text
[ไทย]
"เคสที่ 2 เราทดสอบภาพเซลฟี่เต็มหน้าที่มีอาการผื่นแดงกระจายทั่วใบหน้าครับ

ระบบ DermaScan Face AI วินิจฉัยว่าเป็น โรคโรซาเซีย (Rosacea) ด้วยความมั่นใจ 99.99% เมื่อเปิดดูแบบ Side-by-Side จะเห็นความร้อนของ Grad-CAM กระจายตัวเป็นรูปปีกผีเสื้อที่ชัดเจนมาก ครอบคลุมสันจมูกและสองข้างแก้ม

ในทางการแพทย์ โรซาเซียมักถูกสับสนกับสิวในผู้ใหญ่ แต่จุดต่างสำคัญคือโรซาเซียจะไม่มีสิวอุดตัน (Comedones) แต่จะมีอาการหน้าแดงถาวรและเส้นเลือดฝอยพองตัว ในการทดลอง Step 12 ของเรากับ 224 ภาพ พบว่า 68.1% ของจุดที่โมเดลสนใจในโรคโรซาเซียจะกระจุกตัวอยู่ที่กึ่งกลางใบหน้าเช่นนี้ ซึ่งพิสูจน์ว่าโมเดลเรียนรู้ฟีเจอร์รอยโรคทางการแพทย์อย่างแท้จริงครับ"
```

---

### Case 3: Dermatitis Case $\rightarrow$ Automatic Clinical Caution Note

* **Goal:** Demonstrate correct detection of Dermatitis on a macro lesion while proving the system's honesty and safety via the automated small-sample Caution Note.
* **Exact File Path:**  
  `data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg`
* **Image ID:** `roboflow_face_skin_f9ef0ad817a8` (`split == "test"`)
* **Expected Output on Screen:**
  * **Prediction Badge:** `Dermatitis` (Amber badge)
  * **Calibrated Confidence:** `99.15%`
  * **Input Mode Badge:** `🎯 Macro close-up photo: Face landmarks not detected. Processed 6 skin tile patches.`
  * **Caution Note Banner (Displayed):**  
    `ℹ️ Note: Dermatitis is the smallest class in the training set (93 training / 15 test images; in-domain F1 0.62–0.71) and merges atopic eczema and contact dermatitis. Interpret with extra caution.`
  * **Top-2 Candidates:** `Dermatitis` (99.15%) and `Rosacea` (0.83%)
* **What to Click / Toggle on Screen:**
  1. Highlight the yellow **Caution Note Box** rendered immediately below the prediction badge.
  2. Toggle language to Thai to show the translated caution message:  
     `หมายเหตุ: คลาส Dermatitis มีขนาดชุดข้อมูลฝึกสอนจำกัด (93 ภาพ train / 15 ภาพ test; F1 ในโดเมน 0.62–0.71) และเป็นการรวมระหว่าง Atopic Eczema และ Contact Dermatitis จึงควรระมัดระวังเป็นพิเศษในการแปลผล`
  3. Review the **"When to See a Doctor"** red-flag list (signs of secondary bacterial infection, oozing yellow crusts, intense pruritus).

#### Suggested Presenter Script:

```text
[English]
"Case 3 showcases our commitment to clinical honesty and transparency. 

Here we analyze an eczematous skin lesion. The model correctly identifies Dermatitis with 99.15% confidence. 

However, notice what appears right below the prediction badge: an automated Caution Callout. Because Dermatitis is the smallest class in available smartphone datasets—with only 93 training images—and combines atopic eczema and contact dermatitis, the system proactively alerts the clinician and patient to interpret the result with extra caution. This prevents over-reliance on the AI in lower-data regimes."
```

```text
[ไทย]
"เคสที่ 3 แสดงถึงความซื่อสัตย์และความโปร่งใสของระบบเราครับ

เมื่ออัปโหลดภาพผื่นผิวหนังอักเสบนี้ โมเดลทำนายได้ถูกต้องว่าเป็น โรคผิวหนังอักเสบ (Dermatitis) ด้วยความมั่นใจ 99.15%

แต่จุดที่สำคัญที่สุดคือ แถบคำเตือนสีเหลือง (Caution Note) ที่ปรากฏขึ้นใต้ยี่ห้อผลลัพธ์ทันที เนื่องจากคลาส Dermatitis มีภาพฝึกสอนเพียง 93 ภาพและเกิดจากการรวมระหว่างผื่นภูมิแพ้กับผื่นแพ้สัมผัส ระบบจึงแจ้งเตือนผู้ใช้งานโดยตรงว่าผลลัพธ์นี้มีความไม่แน่นอนสูงกว่าคลาสอื่น และควรพบแพทย์ผิวหนังเสมอ ซึ่งถือเป็นมาตรการความปลอดภัยที่สำคัญมากในทางการแพทย์ครับ"
```

---

### Case 4: Normal Face $\rightarrow$ Healthy Control Caution Note

* **Goal:** Demonstrate healthy skin classification and verify that the system explains the clinical boundary of "Normal" skin without false focal lesion activations.
* **Exact File Path:**  
  `data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg`
* **Image ID:** `skin_assessment_fba55cb1f7a8` (`split == "test"`)
* **Expected Output on Screen:**
  * **Prediction Badge:** `Normal Skin` (Green badge)
  * **Calibrated Confidence:** `100.00%`
  * **Input Mode Badge:** `🎯 Full-face selfie detected: 5 anatomical regions analysed`
  * **Abstention Status:** `abstained = False`
  * **Caution Note Banner (Displayed):**  
    `ℹ️ Note: A 'Normal' result means no strong visual signs of the 3 target conditions (Acne, Rosacea, Dermatitis) were detected. It does NOT guarantee the absence of other skin conditions.`
  * **Top-2 Candidates:** `Normal` (100.0%) and `Dermatitis` (<0.1%)
* **What to Click / Toggle on Screen:**
  1. Inspect the Grad-CAM heatmap: show that unlike Cases 1, 2, and 3, there are **no localized hot spots** of high intensity. The attribution is low and uniform across the entire face.
  2. Point out the **Normal Skin Caution Note**, emphasizing that AI cannot declare a patient completely "disease-free".
  3. Review the general preventive skin health guidance (sun protection, gentle cleansing, barrier hydration).

#### Suggested Presenter Script:

```text
[English]
"In Case 4, we evaluate a healthy control face. The system predicts Normal Skin with 100.00% calibrated confidence.

Looking at the Grad-CAM visualization, you will observe an important phenomenon: there are no localized red clusters or lesion activations anywhere on the face. Across our test benchmarks, healthy faces exhibit uniformly low activation across all five regions.

Equally important is the caution note shown below: 'A Normal result means no strong visual signs of Acne, Rosacea, or Dermatitis were detected. It does NOT guarantee the absence of other skin conditions.' This prevents patients from misinterpreting a negative result as an all-clear for unmodeled pathologies like melanoma or bacterial infections."
```

```text
[ไทย]
"ในเคสที่ 4 เราทดสอบภาพใบหน้าของกลุ่มควบคุมสุขภาพดีครับ ระบบทำนายว่าเป็น ผิวปกติ (Normal Skin) ด้วยความมั่นใจ 100.00%

จุดที่น่าสนใจมากในภาพ Grad-CAM คือ เราจะไม่พบจุดความร้อนกระจุกตัวสีแดงเข้มเหมือนใน 3 เคสแรกเลย การตอบสนองมีความสม่ำเสมอและราบเรียบตลอดทั้ง 5 บริเวณใบหน้า

นอกจากนี้ ระบบยังมีแถบเตือนความปลอดภัยระบุว่า 'ผลผิวปกติหมายถึงการไม่พบสัญญาณของโรคสิว โรซาเซีย หรือผิวหนังอักเสบเท่านั้น ไม่ได้รับประกันการปลอดโรคผิวหนังชนิดอื่น' ซึ่งช่วยป้องกันไม่ให้ผู้ใช้นำผลไปทึกทักเอาเองว่าปลอดจากภาวะอื่นๆ เช่น มะเร็งผิวหนัง หรือการติดเชื้อแบคทีเรียครับ"
```

---

### Case 5: Macro Close-Up (No Face Detected) $\rightarrow$ Automatic Tiling Mode

* **Goal:** Prove that the system handles close-up smartphone photos where facial landmarks cannot be detected, falling back seamlessly to skin tiling.
* **Exact File Path:**  
  `data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg`
* **Image ID:** `roboflow_rosacea_c2e7137b2c96` (`split == "test"`)
* **Expected Output on Screen:**
  * **Prediction Badge:** `Rosacea` (Rose-pink badge)
  * **Calibrated Confidence:** `100.00%`
  * **Input Mode Badge:** `🎯 Macro close-up photo: Face landmarks not detected. Processed 1 skin tile patch.`
  * **Region Table:** Gracefully hidden (no facial regions)
  * **Dominant Region:** `tile_0`
  * **Top-2 Candidates:** `Rosacea` (100.0%) and `Acne` (<0.1%)
  * **Latency:** ~125 ms (fastest execution due to single-patch forward pass)
* **What to Click / Toggle on Screen:**
  1. Note that the input mode badge changes dynamically from "Full-face selfie" to "Macro close-up photo".
  2. Confirm that the 5-region anatomical table is hidden cleanly without JavaScript errors or empty table rows.
  3. Show the single tile Grad-CAM attention map focusing on the telangiectatic erythema.

#### Suggested Presenter Script:

```text
[English]
"In Case 5, we test a macro close-up of a lesion taken without full-face context. 

MediaPipe detects that no full face is present. Rather than failing or rejecting the photo, DermaScan Face AI automatically switches to close-up tiling mode. It verifies that the image contains valid human skin chrominance via our YCrCb and HSV filter, extracts a standard 224x224 patch, and runs the dual-backbone ensemble.

The model diagnoses Rosacea with 100.00% confidence in just 125 milliseconds. The interface adapts gracefully: the 5-region anatomical breakdown table is hidden, and the Grad-CAM heatmap highlights the erythema across the close-up tile. This dual-mode capability is essential for real-world teledermatology."
```

```text
[ไทย]
"ในเคสที่ 5 เป็นภาพถ่ายระยะใกล้ของรอยโรค โดยไม่มีองค์ประกอบของใบหน้าเต็มรูปครับ

เมื่อระบบตรวจพบว่าไม่มีโครงสร้างใบหน้า ระบบจะไม่ตัดการทำงาน แต่จะสลับไปใช้โหมด Macro Close-up Tiling โดยอัตโนมัติ โดยมีฟิลเตอร์ YCrCb และ HSV ตรวจสอบความสมบูรณ์ของพิกเซลผิวหนัง ก่อนดึงภาพแผ่นขนาด 224x224 พิกเซลเข้าสู่โมเดลคู่

ผลลัพธ์วินิจฉัยเป็น Rosacea ด้วยความมั่นใจ 100.00% ภายในเวลาเพียง 125 มิลลิวินาที หน้าเว็บจะปรับการแสดงผลอย่างชาญฉลาดโดยซ่อนตาราง 5 โซนใบหน้าออกไป และแสดงภาพ Grad-CAM บนแผ่นภาพโดยตรง ความยืดหยุ่นสองโหมดนี้ทำให้ระบบสามารถรองรับทั้งภาพเซลฟี่และภาพถ่ายซูมรอยโรคจากผู้ป่วยจริงได้เป็นอย่างดีครับ"
```

---

### Case 6: Ambiguous Lesion $\rightarrow$ Selective Abstention Gate & Triage Mode

* **Goal:** Demonstrate the selective prediction safety feature: when model confidence falls below the validation-derived threshold ($\tau = 0.7852$), the system withholds diagnosis with a Top-2 deferral message. Then demonstrate **Symptomatic 3-Class Triage Mode**.
* **Exact File Path:**  
  `data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg`
* **Image ID:** `roboflow_face_skin_4b730dec0c6d` (`split == "test"`)
* **Expected Output on Screen:**
  * **Abstention Banner (Triggered):**  
    `⚠️ Uncertain — Please Consult a Dermatologist`  
    `Uncertain — please consult a dermatologist. Top-2 candidates: Rosacea (40.5%) and Normal (34.1%).`
  * **Top-2 Candidate Badges:** `Rosacea (40.5%)`, `Normal (34.1%)`
  * **Confidence:** `40.48%` (Below $\tau = 0.7852$)
  * **Probability Bars:** Shows the vertical dashed threshold indicator at 78.5%.
* **What to Click / Toggle on Screen:**
  1. Highlight the prominent amber **Abstention Banner** at the very top of the results panel.
  2. Point out that the confidence of 40.48% is well below the threshold line ($78.52\%$).
  3. Now click the checkbox: **"Symptomatic 3-Class Triage Mode"** on the upload panel, and re-click **"Analyze Photo"**.
  4. Observe how Normal skin probability is driven to **0.0%** via Bayesian prior adjustment, and the probability mass redistributes to `Rosacea` (54.2%) and `Dermatitis` (26.6%).

#### Suggested Presenter Script:

```text
[English]
"Case 6 demonstrates one of our most vital healthcare safety innovations: Selective Abstention.

This image presents a subtle, ambiguous skin condition. The calibrated ensemble yields a top-1 confidence of only 40.48%, which falls far below our validation-derived safety threshold of 78.52%. 

Instead of forcing a high-risk guess, the application triggers our Selective Abstention Banner. It clearly advises: 'Uncertain — please consult a dermatologist,' and reports the two most likely candidate conditions: Rosacea at 40.5% and Normal Skin at 34.1%.

Now, watch what happens when we enable 'Symptomatic 3-Class Triage Mode'. If the clinician knows the patient is presenting with a symptomatic condition, this checkbox invokes Saerens et al. Bayesian prior shift, setting Normal skin prior to zero. The probability instantly rebalances across the three disease classes. This selective prediction mechanism is why our model reaches 99.17% accuracy on retained in-domain cases."
```

```text
[ไทย]
"เคสที่ 6 เป็นการสาธิตนวัตกรรมความปลอดภัยทางการแพทย์ที่สำคัญที่สุด นั่นคือ Selective Abstention (การปฏิเสธการทำนายเมื่อไม่มั่นใจ) ครับ

ภาพนี้เป็นรอยโรคที่มีความคลุมเครือสูง โมเดลคำนวณความมั่นใจสูงสุดได้เพียง 40.48% ซึ่งต่ำกว่าเกณฑ์ความปลอดภัย 78.52% ที่เราคำนวณไว้จากชุด Validation

แทนที่ระบบจะดันทุรังฟันธงคำตอบที่อาจผิดพลาด ระบบจะเปิดแถบแจ้งเตือนสีส้มทันที พร้อมแนะนำให้ปรึกษาแพทย์ผิวหนังอย่างสุภาพ และแจกแจงโรคที่น่าจะเป็น 2 อันดับแรก คือ Rosacea (40.5%) และ Normal (34.1%)

และเมื่อเราเปิดใช้งาน 'Symptomatic 3-Class Triage Mode' ในกรณีที่ทราบแน่ชัดว่าผู้ป่วยมีอาการป่วย ระบบจะใช้การปรับค่าความน่าจะเป็นแบบเบย์เซียน (Bayesian Prior Adjustment) โดยตัดคลาส Normal ออกเป็น 0.0% และกระจายค่าน้ำหนักไปยังกลุ่มโรคผิวหนังทันที กลไกนี้ทำให้ระบบมีความแม่นยำสูงถึง 99.17% ในกรณีที่ระบบมั่นใจครับ"
```

---

### Case 7: Corrupted / Non-Image File $\rightarrow$ Robust HTTP 415 & 400 Handling

* **Goal:** Demonstrate that the web application enforces strict file type and integrity validation without crashing or exposing raw stack traces.
* **Exact File Path:**  
  `data/reports/demo_invalid_inputs/not_an_image.txt` (or upload any text file)
* **Expected Output on Screen:**
  * **HTTP Status Code:** `415 Unsupported Media Type` (or `400 Bad Request` if corrupted)
  * **Alert Box:** Red alert box showing `INVALID_FILE_TYPE`
  * **Error Message (EN):** *"Unsupported file format. Please upload a JPG, PNG, or WEBP image file."*
  * **Error Message (TH):** *"รูปแบบไฟล์ไม่ถูกต้อง รองรับเฉพาะไฟล์รูปภาพ JPG, PNG, หรือ WEBP เท่านั้น"*
* **What to Click / Toggle on Screen:**
  1. Drag and drop `not_an_image.txt` into the drop zone.
  2. Show the friendly red error alert box with the bilingual error explanation.
  3. Upload `corrupted_photo.jpg` to demonstrate HTTP 400 `unreadable_image` handling.

#### Suggested Presenter Script:

```text
[English]
"In Case 7, we test error resilience. We deliberately upload a non-image file, `not_an_image.txt`.

FastAPI immediately intercepts the request, returning HTTP 415 with code `invalid_file_type`. The user receives a clear, friendly bilingual explanation rather than a broken page or raw stack trace. The same graceful recovery applies if corrupted image bytes are sent."
```

```text
[ไทย]
"ในเคสที่ 7 เราทดสอบความทนทานต่อข้อผิดพลาด โดยลองอัปโหลดไฟล์ข้อความ `not_an_image.txt` ครับ

ระบบ FastAPI ตรวจสอบชนิดไฟล์ทันทีและส่งคืนรหัส HTTP 415 พร้อมรหัส `invalid_file_type` หน้าจอจะแสดงกล่องแจ้งเตือนสีแดงพร้อมข้อความอธิบายสองภาษาที่เข้าใจง่าย โดยไม่มีการเปิดเผย Stack Trace ภายใน และเซิร์ฟเวอร์ยังคงทำงานได้อย่างต่อเนื่อง 100% ครับ"
```

---

### Case 8: Non-Skin / Non-Face Image $\rightarrow$ Usable-Skin Filter (HTTP 422)

* **Goal:** Prove that the application rejects images that contain neither a human face nor valid skin chrominance (e.g. solid blue images, landscape photos, household objects).
* **Exact File Path:**  
  `data/reports/demo_invalid_inputs/non_skin_blue.png`
* **Expected Output on Screen:**
  * **HTTP Status Code:** `422 Unprocessable Entity`
  * **Error Badge:** `NO_SKIN_DETECTED`
  * **Error Message (EN):** *"No human face or skin region could be detected. Please ensure the photo clearly shows facial skin under good lighting."*
  * **Error Message (TH):** *"ไม่พบใบหน้าหรือผิวหนังมนุษย์ในรูปภาพ กรุณาถ่ายภาพผิวหน้าให้ชัดเจนในที่มีแสงสว่างเพียงพอ"*
* **What to Click / Toggle on Screen:**
  1. Upload `non_skin_blue.png`.
  2. Point out the HTTP 422 error banner.
  3. Explain the YCrCb/HSV color-space validation rule.

#### Suggested Presenter Script:

```text
[English]
"Finally, in Case 8, we upload an image containing no human face and no skin—in this case, a solid blue square.

Instead of hallucinating a diagnosis on non-skin pixels, our backend service evaluates the image through a color-space chrominance validator in both YCrCb and HSV representations. Finding zero skin pixels, it returns HTTP 422 `no_skin_detected`. 

This prevents false diagnoses on photos of pets, walls, or random objects, ensuring clinical reliability."
```

```text
[ไทย]
"ในเคสสุดท้าย เคสที่ 8 เราทดสอบอัปโหลดภาพที่ไม่ใช่ผิวหนังมนุษย์ เช่น ภาพพื้นหลังสีฟ้าล้วนครับ

แทนที่โมเดลจะทำนายมั่ว (Hallucination) กับสิ่งที่ไม่ใช่ผิวหนัง ระบบของเรามีฟิลเตอร์ตรวจสอบพิกเซลสีผิวในระบบ YCrCb และ HSV ซึ่งเมื่อตรวจไม่พบพิกเซลผิวหนังมนุษย์ ระบบจะส่งคืนรหัส HTTP 422 `no_skin_detected` ทันที

กลไกนี้ช่วยป้องกันไม่ให้ระบบนำภาพสัตว์เลี้ยง เฟอร์นิเจอร์ หรือวัตถุทั่วไปมาแปลผลเป็นโรคผิวหนัง ถือเป็นด่านคัดกรองความปลอดภัยด่านแรกที่สำคัญยิ่งครับ"
```

---

## 5. Summary Table for Live Presenter Reference

| Case # | Scenario | `image_id` | Exact File Path (from project root) | Mode | Patches | Expected Pred | Confidence | Key Feature to Highlight |
|:---:|---|---|---|:---:|:---:|:---:|:---:|---|
| **1** | Full-face selfie $\rightarrow$ Acne | `iga34_65760604461e` | `data/raw/skin-problems-34-on-the-iga-scale/images/169/3.jpg` | `face` | 5 | `acne` | `100.00%` | Papule/comedone Grad-CAM, 5-region table, EN/TH toggle |
| **2** | Full-face selfie $\rightarrow$ Rosacea | `roboflow_rosacea_030fa3e083ad` | `data/raw/roboflow-rosacea/train/rosacea_132_jpg.rf.f192ef0f4f5ca8d1e6240ebb3b373353.jpg` | `face` | 5 | `rosacea` | `99.99%` | Centrofacial malar butterfly flush, differential diagnosis |
| **3** | Dermatitis (Caution Note) | `roboflow_face_skin_f9ef0ad817a8` | `data/raw/roboflow-face-skin/train/eksim/ek58_jpg.rf.91918491ae32a7e484b4613df2139bad.jpg` | `closeup` | 6 | `dermatitis` | `99.15%` | Small-sample caution note, merged eczema explanation |
| **4** | Normal face (Healthy Control) | `skin_assessment_fba55cb1f7a8` | `data/raw/skin-assessment-dataset/skin_assesment_dataset/normal/0_0_caiyilin_0100.jpeg` | `face` | 5 | `normal` | `100.00%` | Diffuse uniform heatmap, clinical boundary of 'Normal' |
| **5** | Close-up (No face detected) | `roboflow_rosacea_c2e7137b2c96` | `data/raw/roboflow-rosacea/test/Rosacea_226_jpg.rf.a226809c359233f00ecacf8fad1de05f.jpg` | `closeup` | 1 | `rosacea` | `100.00%` | Dynamic close-up tiling fallback, 125 ms latency |
| **6** | Low-confidence $\rightarrow$ Abstention | `roboflow_face_skin_4b730dec0c6d` | `data/raw/roboflow-face-skin/train/eksim/ek74_jpg.rf.d1960fee7316eb4dfde286fc4955d153.jpg` | `face` | 5 | `rosacea` | `40.48%` | Amber abstention banner, Top-2 advice, Triage Mode toggle |
| **7** | Corrupted / Non-image upload | `—` | `data/reports/demo_invalid_inputs/not_an_image.txt` | `—` | `—` | `—` | `—` | HTTP 415/400 validation, friendly bilingual alert |
| **8** | Non-skin / Non-face upload | `—` | `data/reports/demo_invalid_inputs/non_skin_blue.png` | `—` | `—` | `—` | `—` | HTTP 422 `no_skin_detected`, color-space skin filter |
