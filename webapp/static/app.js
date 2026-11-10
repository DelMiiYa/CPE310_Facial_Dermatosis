/**
 * DermaScan Face AI — Clinical Frontend Single Page Application
 * CPE310 Healthcare AI System • Bilingual (EN / TH) Interface
 */

const SYSTEM_NAME = "DermaScan Face AI";

// State Management
let currentLang = "en"; // "en" | "th"
let selectedFile = null;
let lastResult = null;
let lastError = null;
let currentImageView = "gradcam"; // "gradcam" | "original" | "sidebyside"

// Constants
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

// Translation Dictionary for Static UI Elements
const UI_TRANSLATIONS = {
  en: {
    system_subtitle: "CPE310 Healthcare AI System — Calibrated Dual-Backbone Facial Dermatosis Classifier",
    lang_toggle_label: "ไทย (TH)",
    banner_disclaimer: "<strong>Educational university demonstration only</strong> — NOT a clinical diagnostic device. Do not use for clinical decision-making. Please consult a qualified dermatologist.",
    banner_privacy: "<strong>Strict Privacy Guarantee:</strong> Uploaded photos are processed strictly in memory and never saved to disk or persistent storage.",
    upload_title: "Photo Input & Analysis",
    upload_badge: "In-Memory",
    drop_primary: "Drag and drop a face photo here, or <strong>browse files</strong>",
    drop_secondary: "Supports JPG, PNG, WEBP (Maximum 10 MB)",
    btn_clear_photo: "Change / Clear Photo",
    triage_title: "Symptomatic 3-Class Triage Mode",
    triage_desc: "Excludes Normal class via Bayesian prior adjustment for known symptomatic lesions",
    btn_analyze: "Analyze Photo",
    btn_analyzing: "Analyzing facial patches & Grad-CAM...",
    btn_mock: "Load Demo Preview (Mock)",
    error_title: "Analysis Request Failed",
    results_title: "Diagnostic Analysis Results",
    abstention_heading: "Uncertain — Please Consult a Dermatologist",
    top2_label: "Top Candidate Conditions:",
    match_lead: "The model's best match is:",
    match_lead_uncertain: "Uncertain — Top candidate:",
    prob_heading: "Class Probabilities (Calibrated Dual-Backbone)",
    threshold_prefix: "Abstention Threshold:",
    explain_heading: "Visual Explainability & Grad-CAM Heatmap",
    explain_caption: "Saliency attention map fusing ResNet-50 (layer4) and EfficientNet-B0 (conv_head)",
    btn_gradcam: "Grad-CAM Heatmap",
    btn_original: "Show Original Image",
    btn_sidebyside: "Side-by-Side",
    split_orig: "Original Photo",
    split_cam: "Grad-CAM Saliency",
    region_table_heading: "Anatomical Region Probability Breakdown",
    th_region: "Region",
    th_top_match: "Top Match",
    ensemble_heading: "Dual-Backbone Ensemble Breakdown",
    models_agree: "Models Agree",
    models_disagree: "Models Disagree — Resolved by Calibrated Soft Fusion",
    info_tag: "Clinical Knowledge Base",
    info_symptoms_title: "Typical Symptoms & Clinical Signs",
    info_distribution_title: "Facial Distribution & Grad-CAM Correlation",
    info_triggers_title: "Common Triggers & Aggravating Factors",
    info_confused_title: "Commonly Confused Conditions (Differential Diagnosis)",
    info_care_title: "General Skin Care Tips (Conservative)",
    info_doctor_title: "When to See a Doctor or Dermatologist",
    info_sources_title: "Reputable Dermatology References",
    limitations_title: "Model Architecture, Validation Benchmarks & Honest Known Limitations",
    arch_title: "Architecture & Dataset",
    perf_title: "In-Domain Test Performance (303 Images)",
    limits_title: "Known Limitations & Clinical Scope",
    err_too_large: "File size exceeds 10 MB limit. Please select a smaller photo.",
    err_invalid_type: "Unsupported file type. Please upload a JPEG, PNG, or WEBP image.",
    err_network: "Network connection error. Failed to reach the analysis service.",
    err_unusable_image: "No clear face or usable skin region detected. Please upload a clear photo of facial skin."
  },
  th: {
    system_subtitle: "ระบบปัญญาประดิษฐ์ทางการแพทย์ CPE310 — โมเดลจำแนกโรคผิวหนังบนใบหน้าแบบ Dual-Backbone พร้อมการปรับเทียบความเชื่อมั่น",
    lang_toggle_label: "English (EN)",
    banner_disclaimer: "<strong>การสาธิตเพื่อการศึกษาในมหาวิทยาลัยเท่านั้น</strong> — มิใช่อุปกรณ์วินิจฉัยทางการแพทย์ ห้ามใช้เพื่อการตัดสินใจทางคลินิก โปรดปรึกษาแพทย์เฉพาะทางด้านผิวหนัง",
    banner_privacy: "<strong>นโยบายความเป็นส่วนตัวเคร่งครัด:</strong> รูปภาพที่อัปโหลดจะถูกประมวลผลในหน่วยความจำชั่วคราว (RAM) เท่านั้น และไม่มีการบันทึกภาพลงในดิสก์",
    upload_title: "นำเข้าภาพถ่าย & วิเคราะห์",
    upload_badge: "ประมวลผลใน RAM",
    drop_primary: "ลากและวางรูปถ่ายใบหน้าตรงนี้ หรือ <strong>คลิกเพื่อเลือกไฟล์</strong>",
    drop_secondary: "รองรับไฟล์ JPG, PNG, WEBP (ขนาดไม่เกิน 10 MB)",
    btn_clear_photo: "เปลี่ยน / ล้างรูปภาพ",
    triage_title: "โหมดคัดกรอง 3 โรคสำหรับผู้มีรอยโรค (Symptomatic Triage)",
    triage_desc: "ตัดคลาสปกติ (Normal) ออกผ่านการปรับความน่าจะเป็นก่อนหน้า (Bayesian prior) สำหรับผู้ที่มีอาการทางผิวหนัง",
    btn_analyze: "วิเคราะห์รูปภาพ",
    btn_analyzing: "กำลังวิเคราะห์รอยโรคและ Grad-CAM...",
    btn_mock: "โหลดตัวอย่างการแสดงผล (Mock)",
    error_title: "การวิเคราะห์ไม่สำเร็จ",
    results_title: "ผลการวิเคราะห์ทางปัญญาประดิษฐ์",
    abstention_heading: "ผลลัพธ์ยังไม่แน่นอน — โปรดปรึกษาแพทย์ผิวหนัง",
    top2_label: "ภาวะที่เป็นไปได้สูงสุด 2 อันดับแรก:",
    match_lead: "ผลลัพธ์ที่ใกล้เคียงที่สุดจากโมเดลคือ:",
    match_lead_uncertain: "ผลลัพธ์ยังไม่แน่นอน — โรคที่มีความน่าจะเป็นสูงสุด:",
    prob_heading: "ความน่าจะเป็นของแต่ละกลุ่มโรค (Calibrated Dual-Backbone)",
    threshold_prefix: "เกณฑ์การงดออกผล (Abstention Threshold):",
    explain_heading: "การอธิบายผลด้วยแผนที่ความร้อน Grad-CAM",
    explain_caption: "แผนที่แสดงจุดที่โมเดลให้ความสนใจ ผสานระหว่าง ResNet-50 (layer4) และ EfficientNet-B0 (conv_head)",
    btn_gradcam: "แผนที่ความร้อน Grad-CAM",
    btn_original: "แสดงภาพถ่ายต้นฉบับ",
    btn_sidebyside: "เปรียบเทียบเคียงข้างกัน",
    split_orig: "ภาพถ่ายต้นฉบับ",
    split_cam: "แผนที่จุดสนใจ Grad-CAM",
    region_table_heading: "รายละเอียดความน่าจะเป็นแยกตามตำแหน่งทางกายวิภาคบนใบหน้า",
    th_region: "ตำแหน่งบนใบหน้า",
    th_top_match: "ผลลัพธ์เด่น",
    ensemble_heading: "รายละเอียดการทำงานร่วมกันของทั้ง 2 โมเดล (Ensemble)",
    models_agree: "โมเดลทั้งสองเห็นพ้องตรงกัน",
    models_disagree: "โมเดลทั้งสองให้ผลต่างกัน — ตัดสินด้วย Calibrated Soft Fusion",
    info_tag: "ฐานความรู้ทางการแพทย์",
    info_symptoms_title: "อาการและรอยโรคที่พบบ่อย",
    info_distribution_title: "การกระจายตัวบนใบหน้าและจุดสนใจของ Grad-CAM",
    info_triggers_title: "ปัจจัยกระตุ้นและสิ่งแวดล้อมที่ทำให้รอยโรคกำเริบ",
    info_confused_title: "ภาวะที่มักสับสนหรือคล้ายคลึงกัน (Differential Diagnosis)",
    info_care_title: "คำแนะนำการดูแลผิวเบื้องต้นแบบประคับประคอง",
    info_doctor_title: "สัญญาณเตือนที่ควรไปพบแพทย์หรือแพทย์ผิวหนัง",
    info_sources_title: "เอกสารอ้างอิงทางการแพทย์ที่น่าเชื่อถือ",
    limitations_title: "สถาปัตยกรรมโมเดล ผลการทดสอบจริง & ข้อจำกัดที่ควรรู้",
    arch_title: "สถาปัตยกรรม & ชุดข้อมูล",
    perf_title: "ประสิทธิภาพบนชุดทดสอบจริงในโดเมน (303 ภาพ)",
    limits_title: "ข้อจำกัดของระบบ & ขอบเขตการทำงาน",
    err_too_large: "ขนาดไฟล์เกิน 10 MB กรุณาเลือกรูปภาพที่มีขนาดเล็กลง",
    err_invalid_type: "รูปแบบไฟล์ไม่รองรับ กรุณาอัปโหลดไฟล์ภาพ JPEG, PNG หรือ WEBP",
    err_network: "เกิดข้อผิดพลาดในการเชื่อมต่อเครือข่าย ไม่สามารถติดต่อระบบวิเคราะห์ได้",
    err_unusable_image: "ตรวจไม่พบใบหน้าหรือผิวหนังที่ชัดเจน กรุณาอัปโหลดภาพผิวหน้าที่มีความคมชัด"
  }
};

// Region name translations
const REGION_NAMES = {
  forehead: { en: "Forehead", th: "หน้าผาก" },
  nose: { en: "Nose", th: "จมูก" },
  left_cheek: { en: "Left Cheek", th: "แก้มซ้าย" },
  right_cheek: { en: "Right Cheek", th: "แก้มขวา" },
  chin: { en: "Chin", th: "คาง" }
};

// Class Labels
const CLASS_LABELS = {
  acne: { en: "Acne", th: "สิว (Acne)" },
  rosacea: { en: "Rosacea", th: "โรคโรซาเซีย (Rosacea)" },
  dermatitis: { en: "Dermatitis", th: "โรคผิวหนังอักเสบ (Dermatitis)" },
  normal: { en: "Normal", th: "ผิวปกติ (Normal)" }
};

// DOM Elements
let el = {};

document.addEventListener("DOMContentLoaded", () => {
  initElements();
  setupEventListeners();
  checkUrlParams();
});

/**
 * Cache DOM Element References
 */
function initElements() {
  el = {
    // Header & Language
    systemName: document.getElementById("system-name"),
    langToggleBtn: document.getElementById("lang-toggle-btn"),
    langLabel: document.getElementById("lang-label"),

    // Upload & Form
    dropZone: document.getElementById("drop-zone"),
    dropZonePrompt: document.getElementById("drop-zone-prompt"),
    fileInput: document.getElementById("file-input"),
    photoPreviewCard: document.getElementById("photo-preview-card"),
    previewImage: document.getElementById("preview-image"),
    previewFilename: document.getElementById("preview-filename"),
    previewFilesize: document.getElementById("preview-filesize"),
    clearPhotoBtn: document.getElementById("clear-photo-btn"),
    triageModeToggle: document.getElementById("triage-mode-toggle"),
    analyzeBtn: document.getElementById("analyze-btn"),
    analyzeSpinner: document.getElementById("analyze-spinner"),
    analyzeBtnText: document.getElementById("analyze-btn-text"),
    mockBtn: document.getElementById("mock-btn"),

    // Error Alert Box
    errorBanner: document.getElementById("error-banner"),
    errorCodeBadge: document.getElementById("error-code-badge"),
    errorMessageText: document.getElementById("error-message-text"),

    // Results Section
    resultsPanel: document.getElementById("results-panel"),
    latencyBadge: document.getElementById("latency-badge"),
    abstentionBanner: document.getElementById("abstention-banner"),
    abstentionHeading: document.getElementById("abstention-heading"),
    abstentionMessage: document.getElementById("abstention-message"),
    abstentionTop2Tags: document.getElementById("abstention-top2-tags"),

    // Primary Prediction
    predictionMatchLead: document.getElementById("prediction-match-lead"),
    predictionBadge: document.getElementById("prediction-badge"),
    predictionConfidence: document.getElementById("prediction-confidence"),
    inputModeBadge: document.getElementById("input-mode-badge"),
    inputModeText: document.getElementById("input-mode-text"),
    cautionNoteBox: document.getElementById("caution-note-box"),
    cautionNoteText: document.getElementById("caution-note-text"),

    // Probability Bars
    probabilityBars: document.getElementById("probability-bars"),
    thresholdIndicatorTag: document.getElementById("threshold-indicator-tag"),
    thresholdLabelText: document.getElementById("threshold-label-text"),

    // Visual Explainability & Images
    btnViewGradcam: document.getElementById("btn-view-gradcam"),
    btnViewOriginal: document.getElementById("btn-view-original"),
    btnViewSidebyside: document.getElementById("btn-view-sidebyside"),
    imageViewerContainer: document.getElementById("image-viewer-container"),
    frameSingle: document.getElementById("frame-single"),
    frameSplit: document.getElementById("frame-split"),
    viewerImgSingle: document.getElementById("viewer-img-single"),
    viewerImgOrig: document.getElementById("viewer-img-orig"),
    viewerImgCam: document.getElementById("viewer-img-cam"),

    // Region Table
    regionTableContainer: document.getElementById("region-table-container"),
    regionTableBody: document.getElementById("region-table-body"),

    // Ensemble Breakdown
    modelBreakdownCard: document.getElementById("model-breakdown-card"),
    modelsAgreeBadge: document.getElementById("models-agree-badge"),
    resnetPredName: document.getElementById("resnet-pred-name"),
    resnetPredConf: document.getElementById("resnet-pred-conf"),
    resnetProbsList: document.getElementById("resnet-probs-list"),
    effnetPredName: document.getElementById("effnet-pred-name"),
    effnetPredConf: document.getElementById("effnet-pred-conf"),
    effnetProbsList: document.getElementById("effnet-probs-list"),
    ensemblePredName: document.getElementById("ensemble-pred-name"),
    ensemblePredConf: document.getElementById("ensemble-pred-conf"),
    ensembleProbsList: document.getElementById("ensemble-probs-list"),

    // Condition Educational Card
    conditionInfoCard: document.getElementById("condition-info-card"),
    infoConditionName: document.getElementById("info-condition-name"),
    infoMatchPhrase: document.getElementById("info-match-phrase"),
    infoShortDesc: document.getElementById("info-short-desc"),
    infoSymptomsList: document.getElementById("info-symptoms-list"),
    infoDistributionText: document.getElementById("info-distribution-text"),
    infoTriggersList: document.getElementById("info-triggers-list"),
    infoConfusedList: document.getElementById("info-confused-list"),
    infoCareList: document.getElementById("info-care-list"),
    infoDoctorList: document.getElementById("info-doctor-list"),
    infoSourcesList: document.getElementById("info-sources-list")
  };
}

/**
 * Register Event Handlers
 */
function setupEventListeners() {
  // Language Switcher
  if (el.langToggleBtn) {
    el.langToggleBtn.addEventListener("click", () => {
      setLanguage(currentLang === "en" ? "th" : "en");
    });
  }

  // File picker trigger
  if (el.dropZone) {
    el.dropZone.addEventListener("click", () => el.fileInput.click());
    el.dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        el.fileInput.click();
      }
    });

    // Drag-and-drop
    ["dragenter", "dragover"].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.remove("dragover");
      });
    });

    el.dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelection(files[0]);
      }
    });
  }

  // File input change
  if (el.fileInput) {
    el.fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  // Clear photo button
  if (el.clearPhotoBtn) {
    el.clearPhotoBtn.addEventListener("click", clearSelectedPhoto);
  }

  // Analyze Button Click
  if (el.analyzeBtn) {
    el.analyzeBtn.addEventListener("click", executeAnalyze);
  }

  // Mock Result Button Click
  if (el.mockBtn) {
    el.mockBtn.addEventListener("click", () => loadMockResult());
  }

  // Segmented Image View Controls
  const viewBtns = [
    { btn: el.btnViewGradcam, view: "gradcam" },
    { btn: el.btnViewOriginal, view: "original" },
    { btn: el.btnViewSidebyside, view: "sidebyside" }
  ];

  viewBtns.forEach(({ btn, view }) => {
    if (btn) {
      btn.addEventListener("click", () => setImageView(view));
    }
  });
}

/**
 * Check URL query parameters (e.g. ?mock=1 or ?lang=th)
 */
function checkUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const langParam = params.get("lang");
  if (langParam === "th" || langParam === "en") {
    setLanguage(langParam);
  }
  if (params.get("mock") === "1") {
    loadMockResult();
  }
}

/**
 * Handle File Selection with Client-Side Validation
 */
function handleFileSelection(file) {
  hideError();
  
  // Extension check
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const isValidType = ALLOWED_MIME_TYPES.has(file.type.toLowerCase()) || ALLOWED_EXTENSIONS.has(ext);

  if (!isValidType) {
    showError({
      error_code: "invalid_file_type",
      detail: UI_TRANSLATIONS.en.err_invalid_type,
      detail_th: UI_TRANSLATIONS.th.err_invalid_type
    });
    clearSelectedPhoto();
    return;
  }

  // File size check (10 MB limit)
  if (file.size > MAX_UPLOAD_BYTES) {
    showError({
      error_code: "file_too_large",
      detail: UI_TRANSLATIONS.en.err_too_large,
      detail_th: UI_TRANSLATIONS.th.err_too_large
    });
    clearSelectedPhoto();
    return;
  }

  selectedFile = file;

  // Display Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    el.previewImage.src = e.target.result;
    el.previewFilename.textContent = file.name;
    el.previewFilesize.textContent = formatBytes(file.size);
    el.photoPreviewCard.classList.remove("hidden");
    el.dropZonePrompt.classList.add("hidden");
    el.analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

/**
 * Clear the selected photo
 */
function clearSelectedPhoto() {
  selectedFile = null;
  el.fileInput.value = "";
  el.previewImage.src = "";
  el.previewFilename.textContent = "";
  el.previewFilesize.textContent = "";
  el.photoPreviewCard.classList.add("hidden");
  el.dropZonePrompt.classList.remove("hidden");
  el.analyzeBtn.disabled = true;
}

/**
 * Send POST /api/analyze to the backend
 */
async function executeAnalyze() {
  if (!selectedFile) return;

  hideError();
  setAnalyzeLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("triage_mode", el.triageModeToggle.checked);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let errData = {};
      try {
        errData = await response.json();
      } catch (_) {
        errData = {
          error_code: `http_${response.status}`,
          detail: `Server returned HTTP ${response.status} ${response.statusText}`,
          detail_th: `เซิร์ฟเวอร์ส่งกลับสถานะ ${response.status} ${response.statusText}`
        };
      }
      showError(errData);
      return;
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    console.error("Network or fetch error:", err);
    showError({
      error_code: "network_error",
      detail: UI_TRANSLATIONS.en.err_network,
      detail_th: UI_TRANSLATIONS.th.err_network
    });
  } finally {
    setAnalyzeLoading(false);
  }
}

/**
 * Load mock demo result (offline fallback support)
 */
async function loadMockResult() {
  hideError();
  setAnalyzeLoading(true);

  const candidateUrls = ["/static/mock_result.json", "./mock_result.json", "mock_result.json"];
  let mockData = null;

  for (const url of candidateUrls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        mockData = await res.json();
        break;
      }
    } catch (_) {
      // try next path
    }
  }

  setAnalyzeLoading(false);

  if (mockData) {
    renderResult(mockData);
  } else {
    showError({
      error_code: "mock_not_found",
      detail: "Could not load mock_result.json from static directory.",
      detail_th: "ไม่สามารถโหลดไฟล์ mock_result.json ได้จากโฟลเดอร์ static"
    });
  }
}

/**
 * Toggle Loading State on Primary Action Button
 */
function setAnalyzeLoading(isLoading) {
  if (isLoading) {
    el.analyzeBtn.disabled = true;
    el.analyzeSpinner.classList.remove("hidden");
    el.analyzeBtnText.textContent = UI_TRANSLATIONS[currentLang].btn_analyzing;
  } else {
    el.analyzeBtn.disabled = !selectedFile;
    el.analyzeSpinner.classList.add("hidden");
    el.analyzeBtnText.textContent = UI_TRANSLATIONS[currentLang].btn_analyze;
  }
}

/**
 * Display Friendly Bilingual Error Alert Box
 */
function showError(errorObj) {
  lastError = errorObj;
  const t = UI_TRANSLATIONS[currentLang];
  const code = errorObj.error_code || "error";
  const message = currentLang === "th" && errorObj.detail_th ? errorObj.detail_th : (errorObj.detail || t.error_title);

  el.errorCodeBadge.textContent = code.toUpperCase();
  el.errorMessageText.textContent = message;
  el.errorBanner.classList.remove("hidden");
  el.errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Hide Error Alert Box
 */
function hideError() {
  lastError = null;
  el.errorBanner.classList.add("hidden");
}

/**
 * Render Complete Results Panel
 */
function renderResult(data) {
  lastResult = data;
  const t = UI_TRANSLATIONS[currentLang];

  // Reveal results panel
  el.resultsPanel.classList.remove("hidden");

  // Latency badge
  if (data.latency_ms !== undefined && data.latency_ms !== null) {
    el.latencyBadge.textContent = `${Number(data.latency_ms).toFixed(1)} ms`;
  }

  // 1. Abstention Banner
  if (data.abstained === true) {
    el.abstentionBanner.classList.remove("hidden");
    el.abstentionHeading.textContent = t.abstention_heading;
    
    // Abstention message
    let msg = data.abstention_message || "";
    if (currentLang === "th") {
      msg = `ความเชื่อมั่นของแบบจำลอง (${(data.confidence * 100).toFixed(1)}%) ต่ำกว่าเกณฑ์ความปลอดภัย (${(data.abstention_threshold * 100).toFixed(1)}%) จึงของดการออกผลลัพธ์แบบเจาะจง โปรดพบแพทย์ผิวหนังเพื่อตรวจวินิจฉัยอย่างละเอียด`;
    }
    el.abstentionMessage.textContent = msg;

    // Top 2 candidates badges
    el.abstentionTop2Tags.innerHTML = "";
    if (Array.isArray(data.top2)) {
      data.top2.forEach((k) => {
        const span = document.createElement("span");
        span.className = "top2-tag";
        const prob = data.probabilities && data.probabilities[k] ? ` (${(data.probabilities[k] * 100).toFixed(1)}%)` : "";
        const label = (CLASS_LABELS[k] && CLASS_LABELS[k][currentLang]) || k;
        span.textContent = `${label}${prob}`;
        el.abstentionTop2Tags.appendChild(span);
      });
    }

    el.predictionMatchLead.textContent = t.match_lead_uncertain;
  } else {
    el.abstentionBanner.classList.add("hidden");
    el.predictionMatchLead.textContent = t.match_lead;
  }

  // 2. Primary Prediction Card
  const predKey = (data.prediction || "normal").toLowerCase();
  const predLabel = (CLASS_LABELS[predKey] && CLASS_LABELS[predKey][currentLang]) || data.prediction_label || data.prediction;
  
  el.predictionBadge.textContent = predLabel;
  el.predictionBadge.className = `prediction-badge badge-${predKey}`;
  el.predictionConfidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;

  // Input Mode details
  if (data.input_mode === "face") {
    const dominantName = formatRegionName(data.dominant_region, currentLang);
    el.inputModeText.textContent = currentLang === "th"
      ? `ตรวจพบใบหน้าเต็มส่วน (Full-face selfie): วิเคราะห์ 5 บริเวณทางกายวิภาค • บริเวณที่โมเดลโฟกัสสูงสุด: ${dominantName}`
      : `Full-face selfie detected: 5 anatomical regions analysed (forehead, nose, left cheek, right cheek, chin) • Dominant region: ${dominantName}`;
  } else {
    const patchCount = data.num_patches || 1;
    const dominantPatch = data.dominant_region || "tile_0";
    el.inputModeText.textContent = currentLang === "th"
      ? `ภาพถ่ายรอยโรคระยะใกล้ (Close-up macro): วิเคราะห์ ${patchCount} แผ่นภาพ • แผ่นภาพที่โฟกัสสูงสุด: ${dominantPatch}`
      : `Close-up macro image: ${patchCount} patches analysed • Dominant patch: ${dominantPatch}`;
  }

  // Caution Note Callout
  if (data.caution_note) {
    el.cautionNoteText.textContent = data.caution_note;
    el.cautionNoteBox.classList.remove("hidden");
  } else {
    el.cautionNoteBox.classList.add("hidden");
  }

  // 3. 4-Class Probability Bars
  renderProbabilityBars(data);

  // 4. Visual Explainability (Images)
  el.viewerImgSingle.src = data.gradcam_image || "";
  el.viewerImgOrig.src = data.original_image || "";
  el.viewerImgCam.src = data.gradcam_image || "";
  setImageView(currentImageView);

  // 5. Per-Region Probability Table
  renderRegionTable(data);

  // 6. Model Ensemble Breakdown
  renderModelBreakdown(data);

  // 7. Condition Educational Knowledge Base Card
  renderConditionInfo(data.info, currentLang);

  // Scroll to results
  el.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Render 4-Class Probability Progress Bars
 */
function renderProbabilityBars(data) {
  const probs = data.probabilities || {};
  const t = UI_TRANSLATIONS[currentLang];
  const threshold = data.abstention_threshold || 0.7852;

  // Threshold Tag
  el.thresholdLabelText.textContent = `${t.threshold_prefix} ${(threshold * 100).toFixed(1)}%`;

  // Sort classes by probability descending
  const sortedClasses = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  el.probabilityBars.innerHTML = "";

  sortedClasses.forEach(([clsKey, prob]) => {
    const row = document.createElement("div");
    row.className = "prob-row";

    const labelRow = document.createElement("div");
    labelRow.className = "prob-label-row";

    const nameGroup = document.createElement("div");
    nameGroup.className = "prob-name-group";

    const square = document.createElement("span");
    square.className = `prob-indicator-square fill-${clsKey}`;

    const name = document.createElement("span");
    name.className = "prob-name";
    name.textContent = (CLASS_LABELS[clsKey] && CLASS_LABELS[clsKey][currentLang]) || clsKey;

    nameGroup.appendChild(square);
    nameGroup.appendChild(name);

    const val = document.createElement("span");
    val.className = "prob-value";
    val.textContent = `${(prob * 100).toFixed(2)}%`;

    labelRow.appendChild(nameGroup);
    labelRow.appendChild(val);

    const track = document.createElement("div");
    track.className = "prob-track";

    // Marker line for abstention threshold
    const marker = document.createElement("div");
    marker.className = "threshold-marker-line";
    marker.style.left = `${Math.min(threshold * 100, 100)}%`;
    track.appendChild(marker);

    const fill = document.createElement("div");
    fill.className = `prob-fill fill-${clsKey}`;
    fill.style.width = "0%";
    track.appendChild(fill);

    row.appendChild(labelRow);
    row.appendChild(track);
    el.probabilityBars.appendChild(row);

    // Animate fill smoothly
    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.width = `${Math.min(prob * 100, 100)}%`;
      }, 50);
    });
  });
}

/**
 * Switch Image Viewer Display Mode
 */
function setImageView(mode) {
  currentImageView = mode;

  [el.btnViewGradcam, el.btnViewOriginal, el.btnViewSidebyside].forEach((btn) => {
    if (btn) btn.classList.remove("active");
  });

  if (mode === "gradcam") {
    el.btnViewGradcam.classList.add("active");
    el.frameSingle.classList.remove("hidden");
    el.frameSplit.classList.add("hidden");
    if (lastResult && lastResult.gradcam_image) {
      el.viewerImgSingle.src = lastResult.gradcam_image;
    }
  } else if (mode === "original") {
    el.btnViewOriginal.classList.add("active");
    el.frameSingle.classList.remove("hidden");
    el.frameSplit.classList.add("hidden");
    if (lastResult && lastResult.original_image) {
      el.viewerImgSingle.src = lastResult.original_image;
    }
  } else if (mode === "sidebyside") {
    el.btnViewSidebyside.classList.add("active");
    el.frameSingle.classList.add("hidden");
    el.frameSplit.classList.remove("hidden");
  }
}

/**
 * Render Per-Region Probability Table
 */
function renderRegionTable(data) {
  if (data.input_mode !== "face" || !data.region_probabilities) {
    el.regionTableContainer.classList.add("hidden");
    return;
  }

  el.regionTableContainer.classList.remove("hidden");
  el.regionTableBody.innerHTML = "";

  const regionOrder = ["forehead", "nose", "left_cheek", "right_cheek", "chin"];
  const dominant = data.dominant_region || "";

  regionOrder.forEach((rKey) => {
    const rProbs = data.region_probabilities[rKey];
    if (!rProbs) return;

    // Determine top class for this region
    let topClass = "acne";
    let maxP = -1;
    for (const [c, p] of Object.entries(rProbs)) {
      if (p > maxP) {
        maxP = p;
        topClass = c;
      }
    }

    const tr = document.createElement("tr");
    if (rKey === dominant) {
      tr.className = "dominant-row";
    }

    // Region name cell
    const tdRegion = document.createElement("td");
    tdRegion.className = "region-name-cell";
    tdRegion.innerHTML = `<span>${formatRegionName(rKey, currentLang)}</span>`;
    if (rKey === dominant) {
      tdRegion.innerHTML += `<span class="dominant-tag" title="Highest Grad-CAM Activation">Dominant</span>`;
    }

    // Top match cell
    const tdTop = document.createElement("td");
    const topLabel = (CLASS_LABELS[topClass] && CLASS_LABELS[topClass][currentLang]) || topClass;
    tdTop.innerHTML = `<strong>${topLabel}</strong> (${(maxP * 100).toFixed(1)}%)`;

    // 4 classes percentages
    const tdAcne = document.createElement("td");
    tdAcne.textContent = `${((rProbs.acne || 0) * 100).toFixed(1)}%`;

    const tdRosacea = document.createElement("td");
    tdRosacea.textContent = `${((rProbs.rosacea || 0) * 100).toFixed(1)}%`;

    const tdDerm = document.createElement("td");
    tdDerm.textContent = `${((rProbs.dermatitis || 0) * 100).toFixed(1)}%`;

    const tdNormal = document.createElement("td");
    tdNormal.textContent = `${((rProbs.normal || 0) * 100).toFixed(1)}%`;

    tr.appendChild(tdRegion);
    tr.appendChild(tdTop);
    tr.appendChild(tdAcne);
    tr.appendChild(tdRosacea);
    tr.appendChild(tdDerm);
    tr.appendChild(tdNormal);

    el.regionTableBody.appendChild(tr);
  });
}

/**
 * Render Model Ensemble Comparison Breakdown
 */
function renderModelBreakdown(data) {
  const mb = data.model_breakdown;
  const t = UI_TRANSLATIONS[currentLang];

  if (!mb) {
    el.modelBreakdownCard.classList.add("hidden");
    return;
  }

  el.modelBreakdownCard.classList.remove("hidden");

  // Models Agree Badge
  if (mb.models_agree) {
    el.modelsAgreeBadge.textContent = t.models_agree;
    el.modelsAgreeBadge.className = "badge badge-success";
  } else {
    el.modelsAgreeBadge.textContent = t.models_disagree;
    el.modelsAgreeBadge.className = "badge badge-warning";
  }

  // ResNet-50
  if (mb.resnet50) {
    const resPred = mb.resnet50.prediction || "--";
    el.resnetPredName.textContent = (CLASS_LABELS[resPred] && CLASS_LABELS[resPred][currentLang]) || resPred;
    el.resnetPredConf.textContent = `${((mb.resnet50.confidence || 0) * 100).toFixed(1)}%`;
    renderMiniProbs(el.resnetProbsList, mb.resnet50.probabilities);
  }

  // EfficientNet-B0
  if (mb.efficientnet_b0) {
    const effPred = mb.efficientnet_b0.prediction || "--";
    el.effnetPredName.textContent = (CLASS_LABELS[effPred] && CLASS_LABELS[effPred][currentLang]) || effPred;
    el.effnetPredConf.textContent = `${((mb.efficientnet_b0.confidence || 0) * 100).toFixed(1)}%`;
    renderMiniProbs(el.effnetProbsList, mb.efficientnet_b0.probabilities);
  }

  // Calibrated Fusion Ensemble
  const ensPred = data.prediction || "--";
  el.ensemblePredName.textContent = (CLASS_LABELS[ensPred] && CLASS_LABELS[ensPred][currentLang]) || ensPred;
  el.ensemblePredConf.textContent = `${((data.confidence || 0) * 100).toFixed(1)}%`;
  renderMiniProbs(el.ensembleProbsList, data.probabilities);
}

function renderMiniProbs(container, probsObj) {
  if (!container || !probsObj) return;
  container.innerHTML = "";
  for (const [k, v] of Object.entries(probsObj)) {
    const row = document.createElement("div");
    row.className = "mini-prob-row";
    const name = (CLASS_LABELS[k] && CLASS_LABELS[k][currentLang]) || k;
    row.innerHTML = `<span>${name}</span><span>${(v * 100).toFixed(1)}%</span>`;
    container.appendChild(row);
  }
}

/**
 * Render Condition Educational Knowledge Base Card
 */
function renderConditionInfo(infoObj, lang) {
  if (!infoObj) return;

  // Use localized sub-object if available
  const content = (infoObj[lang] && typeof infoObj[lang] === "object") ? infoObj[lang] : infoObj;

  el.infoConditionName.textContent = content.name || "";
  el.infoMatchPhrase.textContent = content.model_match_phrase || "";
  el.infoShortDesc.textContent = content.short_description || "";

  // 1. Symptoms
  populateList(el.infoSymptomsList, content.typical_symptoms);

  // 2. Facial Distribution & Grad-CAM
  el.infoDistributionText.textContent = content.typical_facial_distribution || "";

  // 3. Triggers
  populateList(el.infoTriggersList, content.common_triggers_or_factors);

  // 4. Commonly Confused With (Differential Diagnosis)
  el.infoConfusedList.innerHTML = "";
  if (Array.isArray(content.commonly_confused_with)) {
    content.commonly_confused_with.forEach((item) => {
      const div = document.createElement("div");
      div.className = "differential-item";
      div.innerHTML = `<strong>${item.condition}</strong><p>${item.how_it_differs}</p>`;
      el.infoConfusedList.appendChild(div);
    });
  }

  // 5. Care Tips
  populateList(el.infoCareList, content.general_care_tips);

  // 6. See a Doctor If
  populateList(el.infoDoctorList, content.see_a_doctor_if);

  // 7. Medical Sources
  const sources = infoObj.sources || content.sources || [];
  el.infoSourcesList.innerHTML = "";
  if (Array.isArray(sources)) {
    sources.forEach((s) => {
      const a = document.createElement("a");
      a.className = "source-link-item";
      a.href = s.url;
      a.textContent = s.title;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      el.infoSourcesList.appendChild(a);
    });
  }
}

function populateList(container, items) {
  if (!container) return;
  container.innerHTML = "";
  if (Array.isArray(items)) {
    items.forEach((txt) => {
      const li = document.createElement("li");
      li.textContent = txt;
      container.appendChild(li);
    });
  }
}

/**
 * Language Switching Functionality (EN / TH)
 */
function setLanguage(lang) {
  currentLang = lang;
  document.documentElement.lang = lang;
  const t = UI_TRANSLATIONS[lang];

  // Update Language Toggle Button
  if (el.langLabel) {
    el.langLabel.textContent = t.lang_toggle_label;
  }

  // Update all data-i18n attributes in the document
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    if (t[key]) {
      node.innerHTML = t[key];
    }
  });

  // Re-render active error banner if present
  if (lastError) {
    showError(lastError);
  }

  // Re-render full result view with new language immediately
  if (lastResult) {
    renderResult(lastResult);
  }
}

/**
 * Helper: Format Region Names
 */
function formatRegionName(regionKey, lang) {
  if (!regionKey) return "";
  const key = regionKey.toLowerCase();
  if (REGION_NAMES[key] && REGION_NAMES[key][lang]) {
    return REGION_NAMES[key][lang];
  }
  return regionKey.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Helper: Format Bytes to human readable string
 */
function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}
