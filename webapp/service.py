"""DermaScan Face AI — Service Layer.

Provides thread-safe, single-copy GPU/CPU inference, confidence calibration,
Bayes prior adjustment (triage mode), Grad-CAM explainability, and the frozen
/api/analyze contract response dictionary.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    apply_bayes_prior_adjustment,
    predict_calibrated_ensemble,
    temperature_scale_probs,
)
from src.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, get_eval_transforms
from src.detection import (
    FaceDetector,
    FaceLandmarker,
    extract_face_patches,
    extract_tile_patches,
)
from src.ensemble import DermatosisEnsemble
from src.explainability import get_target_layers

logger = logging.getLogger(__name__)

# ==============================================================================
# Frozen System Constants & Calibration Parameters
# ==============================================================================

SYSTEM_NAME: str = "DermaScan Face AI"

CALIBRATION_T_RESNET: float = 0.60
CALIBRATION_T_EFFNET: float = 1.20
CALIBRATION_W_RESNET: float = 0.40
CALIBRATION_W_EFFNET: float = 0.60

# Abstention threshold derived from validation split (294 images, runs/ensemble_seed42/val_predictions.csv)
# by scripts/pick_abstention_threshold.py at 85.0% coverage:
# - Coverage: 85.0% (250/294 retained, 44 abstained)
# - Validation Accuracy: 99.20%
# - Validation Macro-F1: 0.9940
# - Threshold (tau): 0.7852
ABSTENTION_THRESHOLD: float = 0.7852

# Source distribution class frequencies from training manifest (1617 total training images):
# Acne: 591, Rosacea: 466, Dermatitis: 63, Normal: 497
TRAINING_SOURCE_PRIORS: np.ndarray = np.array(
    [591.0 / 1617.0, 466.0 / 1617.0, 63.0 / 1617.0, 497.0 / 1617.0],
    dtype=np.float64,
)

# Symptomatic 3-class triage prior (excluding Normal):
TRIAGE_TARGET_PRIORS: np.ndarray = np.array(
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
    dtype=np.float64,
)

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES: Set[str] = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
FACE_REGIONS: List[str] = ["forehead", "nose", "left_cheek", "right_cheek", "chin"]

DISCLAIMER_TEXT: str = (
    "Educational research demonstration only — NOT a medical diagnosis. "
    "Do not use for clinical decision-making. Please consult a qualified dermatologist."
)


# ==============================================================================
# Custom Exceptions for Friendly Error Handling
# ==============================================================================

class WebAppInputError(ValueError):
    """Exception raised for client input errors conforming to the frozen error contract."""

    def __init__(self, detail: str, detail_th: str, error_code: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.detail_th = detail_th
        self.error_code = error_code
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detail": self.detail,
            "detail_th": self.detail_th,
            "error_code": self.error_code,
        }


# ==============================================================================
# Usable Skin Region Check (Macro Close-up Rejection)
# ==============================================================================

def has_usable_skin_region(img_bgr: np.ndarray) -> Tuple[bool, float]:
    """Deterministically check if an image contains usable human skin / erythema content.

    Used when face_detected is False (macro close-up mode) to reject blank images,
    screenshots, landscapes, and other non-dermatological inputs.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return False, 0.0
    if len(img_bgr.shape) != 3 or min(img_bgr.shape[:2]) < 64:
        return False, 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_std = float(np.std(gray))
    if gray_std < 8.0:
        return False, 0.0

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask_ycrcb = (y >= 20) & (y <= 245) & (cr >= 125) & (cr <= 188) & (cb >= 70) & (cb <= 145)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask_hsv_sat = (s >= 12) & (s <= 230) & (v >= 25)
    mask_hsv_hue = (h <= 32) | (h >= 155)

    skin_mask = mask_ycrcb & mask_hsv_sat & mask_hsv_hue
    skin_frac = float(np.mean(skin_mask))
    return (skin_frac >= 0.15, skin_frac)


# ==============================================================================
# Helper Formatting Functions
# ==============================================================================

def _format_probabilities(probs: np.ndarray) -> Dict[str, float]:
    """Convert probability array to lowercase class dict and ensure exact 1.0 sum."""
    d = {CLASS_NAMES[i].lower(): round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
    diff = round(1.0 - sum(d.values()), 4)
    if abs(diff) > 0 and abs(diff) < 0.005:
        max_k = max(d, key=d.get)
        d[max_k] = round(d[max_k] + diff, 4)
    return d


def _encode_png_base64(img_bgr: np.ndarray, max_dim: int = 900) -> str:
    """Encode BGR image to PNG base64 data URL, downsampling if larger than max_dim."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format")
    b64_str = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


# ==============================================================================
# WebAppDermatosisService
# ==============================================================================

class WebAppDermatosisService:
    """Inference and explainability service for DermaScan Face AI.

    Loads ResNet-50, EfficientNet-B0, and MediaPipe detectors ONCE, sharing the
    backbones between inference and Grad-CAM to stay safely within the 4 GB VRAM budget.
    A re-entrant threading lock guarantees sequential, thread-safe request processing.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self._lock = threading.Lock()

        # Select compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing WebAppDermatosisService on device: {self.device}")

        # 1. Load Ensemble ONCE
        self.ensemble = DermatosisEnsemble(
            weights=(CALIBRATION_W_RESNET, CALIBRATION_W_EFFNET),
            device=self.device,
        )

        # 2. Enable gradient tracking on backbones so Grad-CAM can use the exact same instances
        self.ensemble.resnet.requires_grad_(True)
        self.ensemble.effnet.requires_grad_(True)

        # 3. Attach GradCAM directly to ensemble backbones
        resnet_target_layers = get_target_layers(self.ensemble.resnet)
        effnet_target_layers = get_target_layers(self.ensemble.effnet)
        self.cam_resnet = GradCAM(model=self.ensemble.resnet, target_layers=resnet_target_layers)
        self.cam_effnet = GradCAM(model=self.ensemble.effnet, target_layers=effnet_target_layers)

        # 4. MediaPipe detectors
        self.detector = FaceDetector()
        self.landmarker = FaceLandmarker()

        # 5. Transforms
        self.eval_transforms = get_eval_transforms(img_size=224)

        # 6. Knowledge base cache
        self._knowledge_base: Dict[str, Any] = self._load_knowledge_base()

        logger.info("WebAppDermatosisService initialization complete.")

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load webapp/knowledge_base.json if present; otherwise returns empty dict."""
        kb_path = PROJECT_ROOT / "webapp" / "knowledge_base.json"
        if kb_path.exists():
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load knowledge base from {kb_path}: {e}")
                return {}
        return {}

    def get_knowledge_base_entry(self, condition: str) -> Dict[str, Any]:
        """Retrieve condition metadata, reloading from disk if file was created after service start."""
        if not self._knowledge_base:
            self._knowledge_base = self._load_knowledge_base()
        return self._knowledge_base.get(condition.lower(), {})

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()

    def analyze(self, image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]:
        """Execute full end-to-end inference, calibration, and Grad-CAM explainability pipeline.

        Args:
            image_bytes: Raw binary image payload.
            triage_mode: If True, applies Saerens et al. (2002) Bayesian prior shift to
                         re-weight probabilities over the 3 symptomatic conditions (zeroing Normal).

        Returns:
            Dictionary strictly conforming to the frozen /api/analyze JSON contract.
        """
        t0 = time.perf_counter()

        # Input validation
        if not image_bytes or len(image_bytes) == 0:
            raise WebAppInputError(
                detail="The uploaded file is empty. Please upload a valid image.",
                detail_th="ไฟล์ที่อัปโหลดว่างเปล่า กรุณาอัปโหลดรูปภาพที่ถูกต้อง",
                error_code="empty_file",
                status_code=400,
            )

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise WebAppInputError(
                detail=f"Uploaded image size ({len(image_bytes)} bytes) exceeds the maximum allowed 10 MB.",
                detail_th="ขนาดไฟล์รูปภาพเกินกำหนดสูงสุด 10 MB",
                error_code="file_too_large",
                status_code=413,
            )

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            raise WebAppInputError(
                detail="Could not decode image file. Please provide a valid JPEG, PNG, or WebP image.",
                detail_th="ไม่สามารถอ่านไฟล์รูปภาพได้ กรุณาใช้ไฟล์ JPEG, PNG หรือ WebP ที่ถูกต้อง",
                error_code="unreadable_image",
                status_code=400,
            )

        # Thread-safe GPU execution
        with self._lock:
            # 1. Face detection
            det_info = self.detector.detect(img_bgr)
            is_face = bool(det_info.get("face_detected", False))

            if is_face:
                face_res = extract_face_patches(
                    img_bgr,
                    detection_info=det_info,
                    landmarker=self.landmarker,
                    target_size=224,
                )
                patch_bgrs = [face_res["patches"][r] for r in FACE_REGIONS]
                patch_names = list(FACE_REGIONS)
                boxes = [face_res["boxes"][r] for r in FACE_REGIONS]
                canvas_bgr = img_bgr
                input_mode = "face"
            else:
                ok_skin, skin_frac = has_usable_skin_region(img_bgr)
                if not ok_skin:
                    raise WebAppInputError(
                        detail=(
                            "No human face or usable skin lesion region was detected in the image. "
                            "Please provide a clear facial photo or skin close-up."
                        ),
                        detail_th="ไม่พบใบหน้าหรือบริเวณรอยโรคผิวหนังที่สามารถวิเคราะห์ได้ กรุณาอัปโหลดภาพใบหน้าหรือภาพระยะใกล้ของผิวหนังที่ชัดเจน",
                        error_code="no_skin_detected",
                        status_code=422,
                    )

                tile_res = extract_tile_patches(img_bgr, target_size=224, max_tiles=9)
                patch_bgrs = tile_res["patches"]
                boxes = tile_res["boxes"]
                canvas_bgr = tile_res["image_tiled"]
                input_mode = "closeup"
                patch_names = [f"tile_{i}" for i in range(len(patch_bgrs))]

                if len(patch_bgrs) == 0:
                    raise WebAppInputError(
                        detail="Could not extract patches from the skin image.",
                        detail_th="ไม่สามารถแยกส่วนภาพผิวหนังได้",
                        error_code="no_skin_detected",
                        status_code=422,
                    )

            # 2. Build stacked tensor for all patches
            patch_tensors = [
                self.eval_transforms(cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB))
                for p_bgr in patch_bgrs
            ]
            stacked_tensors = torch.stack(patch_tensors, dim=0)
            k_patches = len(patch_bgrs)

            # 3. Model inference pass under @torch.no_grad()
            ens_raw = self.ensemble.predict_image_from_patches(stacked_tensors, aggregation="mean")
            r_img_probs = ens_raw["resnet_probs"].detach().cpu().numpy().astype(np.float64)
            e_img_probs = ens_raw["effnet_probs"].detach().cpu().numpy().astype(np.float64)
            r_patch_probs = ens_raw["resnet_patch_probs"].detach().cpu().numpy().astype(np.float64)
            e_patch_probs = ens_raw["effnet_patch_probs"].detach().cpu().numpy().astype(np.float64)

            # 4. Temperature-scaled calibrated probabilities
            cal_img_probs = predict_calibrated_ensemble(
                r_img_probs[np.newaxis, :],
                e_img_probs[np.newaxis, :],
                w_resnet=CALIBRATION_W_RESNET,
                t_resnet=CALIBRATION_T_RESNET,
                t_effnet=CALIBRATION_T_EFFNET,
            )[0]

            cal_patch_probs = predict_calibrated_ensemble(
                r_patch_probs,
                e_patch_probs,
                w_resnet=CALIBRATION_W_RESNET,
                t_resnet=CALIBRATION_T_RESNET,
                t_effnet=CALIBRATION_T_EFFNET,
            )

            # Calibrated individual backbone probabilities
            scaled_res = temperature_scale_probs(r_img_probs[np.newaxis, :], CALIBRATION_T_RESNET)[0]
            scaled_eff = temperature_scale_probs(e_img_probs[np.newaxis, :], CALIBRATION_T_EFFNET)[0]

            # 5. Symptomatic Triage Prior Adjustment (Saerens et al. 2002)
            if triage_mode:
                cal_img_probs = apply_bayes_prior_adjustment(
                    cal_img_probs[np.newaxis, :],
                    TRAINING_SOURCE_PRIORS,
                    TRIAGE_TARGET_PRIORS,
                )[0]
                cal_patch_probs = apply_bayes_prior_adjustment(
                    cal_patch_probs,
                    TRAINING_SOURCE_PRIORS,
                    TRIAGE_TARGET_PRIORS,
                )
                scaled_res = apply_bayes_prior_adjustment(
                    scaled_res[np.newaxis, :],
                    TRAINING_SOURCE_PRIORS,
                    TRIAGE_TARGET_PRIORS,
                )[0]
                scaled_eff = apply_bayes_prior_adjustment(
                    scaled_eff[np.newaxis, :],
                    TRAINING_SOURCE_PRIORS,
                    TRIAGE_TARGET_PRIORS,
                )[0]

            # 6. Rank predictions and determine confidence
            sorted_idxs = np.argsort(-cal_img_probs)
            pred_idx = int(sorted_idxs[0])
            prediction_label = CLASS_NAMES[pred_idx]
            prediction = prediction_label.lower()

            top2_idx0 = int(sorted_idxs[0])
            top2_idx1 = int(sorted_idxs[1])
            top2 = [CLASS_NAMES[top2_idx0].lower(), CLASS_NAMES[top2_idx1].lower()]

            formatted_probs = _format_probabilities(cal_img_probs)
            confidence = formatted_probs[prediction]

            # Abstention evaluation
            abstained = bool(confidence < ABSTENTION_THRESHOLD)
            if abstained:
                abstention_message = (
                    f"Uncertain — please consult a dermatologist. Top-2 candidates: "
                    f"{CLASS_NAMES[top2_idx0]} ({cal_img_probs[top2_idx0] * 100:.1f}%) and "
                    f"{CLASS_NAMES[top2_idx1]} ({cal_img_probs[top2_idx1] * 100:.1f}%)."
                )
            else:
                abstention_message = None

            # Caution note
            caution_note = None
            if prediction == "dermatitis":
                caution_note = (
                    "Note: Dermatitis is the smallest class in the training set (93 training / 15 test images; "
                    "in-domain F1 0.62–0.71) and merges atopic eczema and contact dermatitis. Interpret with extra caution."
                )
            elif prediction == "normal":
                caution_note = (
                    "Note: A 'Normal' result means no strong visual signs of the 3 target conditions (Acne, Rosacea, Dermatitis) "
                    "were detected. It does NOT guarantee the absence of other skin conditions."
                )

            # 7. Grad-CAM saliency computation for target class pred_idx
            x_grad = stacked_tensors.to(self.ensemble.device, dtype=torch.float32)
            targets = [ClassifierOutputTarget(pred_idx)] * k_patches

            with torch.enable_grad():
                resnet_cams = self.cam_resnet(input_tensor=x_grad, targets=targets)
                effnet_cams = self.cam_effnet(input_tensor=x_grad, targets=targets)

            # Fuse patch CAMs (0.5 ResNet + 0.5 EffNet) and normalize to [0, 1]
            ens_cams = np.zeros_like(resnet_cams)
            for k in range(k_patches):
                raw_fuse = 0.5 * resnet_cams[k] + 0.5 * effnet_cams[k]
                f_min = float(raw_fuse.min())
                f_max = float(raw_fuse.max())
                if f_max > f_min + 1e-6:
                    ens_cams[k] = (raw_fuse - f_min) / (f_max - f_min)
                else:
                    ens_cams[k] = raw_fuse

            # Identify dominant patch
            patch_means = [float(np.mean(ens_cams[k])) for k in range(k_patches)]
            dominant_region = patch_names[int(np.argmax(patch_means))]

            # Project patch heatmaps back onto full image canvas
            cH, cW = canvas_bgr.shape[:2]
            full_cam_ens = np.zeros((cH, cW), dtype=np.float32)
            full_counts = np.zeros((cH, cW), dtype=np.float32)

            for k, (x1, y1, x2, y2) in enumerate(boxes):
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                if x2 > x1 and y2 > y1:
                    ens_cam_res = cv2.resize(ens_cams[k], (bw, bh), interpolation=cv2.INTER_LINEAR)
                    full_cam_ens[y1:y2, x1:x2] += ens_cam_res
                    full_counts[y1:y2, x1:x2] += 1.0

            mask = full_counts > 0
            full_cam_ens[mask] /= full_counts[mask]

            c_min = float(full_cam_ens.min())
            c_max = float(full_cam_ens.max())
            if c_max > c_min + 1e-6:
                full_cam_ens = (full_cam_ens - c_min) / (c_max - c_min)

            canvas_rgb_float = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            full_overlay_rgb = show_cam_on_image(canvas_rgb_float, full_cam_ens, use_rgb=True)

            # Cleanup gradient memory
            del x_grad
            del resnet_cams
            del effnet_cams
            self.ensemble.resnet.zero_grad(set_to_none=True)
            self.ensemble.effnet.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 8. Base64 PNG Encoding (downsampling if max dimension > 900)
            overlay_bgr = cv2.cvtColor(full_overlay_rgb, cv2.COLOR_RGB2BGR)
            if max(cH, cW) > 900:
                scale = 900.0 / float(max(cH, cW))
                new_w = max(1, int(round(cW * scale)))
                new_h = max(1, int(round(cH * scale)))
                orig_to_encode = cv2.resize(canvas_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cam_to_encode = cv2.resize(overlay_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                orig_to_encode = canvas_bgr
                cam_to_encode = overlay_bgr

            _, orig_buf = cv2.imencode(".png", orig_to_encode)
            original_image_b64 = f"data:image/png;base64,{base64.b64encode(orig_buf).decode('ascii')}"

            _, cam_buf = cv2.imencode(".png", cam_to_encode)
            gradcam_image_b64 = f"data:image/png;base64,{base64.b64encode(cam_buf).decode('ascii')}"

            # 9. Region probabilities (face mode only; None in closeup mode)
            if input_mode == "face":
                region_probabilities: Optional[Dict[str, Dict[str, float]]] = {
                    region: _format_probabilities(cal_patch_probs[k])
                    for k, region in enumerate(FACE_REGIONS)
                }
            else:
                region_probabilities = None

            # 10. Model breakdown
            r_pred_idx = int(np.argmax(scaled_res))
            e_pred_idx = int(np.argmax(scaled_eff))
            r_probs_fmt = _format_probabilities(scaled_res)
            e_probs_fmt = _format_probabilities(scaled_eff)

            model_breakdown = {
                "resnet50": {
                    "prediction": CLASS_NAMES[r_pred_idx].lower(),
                    "confidence": r_probs_fmt[CLASS_NAMES[r_pred_idx].lower()],
                    "probabilities": r_probs_fmt,
                },
                "efficientnet_b0": {
                    "prediction": CLASS_NAMES[e_pred_idx].lower(),
                    "confidence": e_probs_fmt[CLASS_NAMES[e_pred_idx].lower()],
                    "probabilities": e_probs_fmt,
                },
                "models_agree": bool(r_pred_idx == e_pred_idx),
                "calibration_params": {
                    "t_resnet": CALIBRATION_T_RESNET,
                    "t_effnet": CALIBRATION_T_EFFNET,
                    "w_resnet": CALIBRATION_W_RESNET,
                    "w_effnet": CALIBRATION_W_EFFNET,
                },
            }

            # 11. Retrieve condition knowledge base entry
            info = self.get_knowledge_base_entry(prediction)

            # 12. Calculate latency
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            # Assemble frozen contract dictionary
            return {
                "prediction": prediction,
                "prediction_label": prediction_label,
                "confidence": confidence,
                "abstained": abstained,
                "abstention_threshold": ABSTENTION_THRESHOLD,
                "abstention_message": abstention_message,
                "caution_note": caution_note,
                "triage_mode": triage_mode,
                "probabilities": formatted_probs,
                "top2": top2,
                "input_mode": input_mode,
                "face_detected": is_face,
                "num_patches": k_patches,
                "dominant_region": dominant_region,
                "region_probabilities": region_probabilities,
                "model_breakdown": model_breakdown,
                "original_image": original_image_b64,
                "gradcam_image": gradcam_image_b64,
                "latency_ms": latency_ms,
                "info": info,
                "disclaimer": DISCLAIMER_TEXT,
            }


# ==============================================================================
# Module Singleton Accessor & Top-Level Convenience Function
# ==============================================================================

_SERVICE_INSTANCE: Optional[WebAppDermatosisService] = None
_SERVICE_INIT_LOCK = threading.Lock()


def get_service(device: Optional[Union[str, torch.device]] = None) -> WebAppDermatosisService:
    """Return the thread-safe singleton WebAppDermatosisService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        with _SERVICE_INIT_LOCK:
            if _SERVICE_INSTANCE is None:
                _SERVICE_INSTANCE = WebAppDermatosisService(device=device)
    return _SERVICE_INSTANCE


def analyze(image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]:
    """Analyze image bytes using the singleton WebAppDermatosisService instance."""
    return get_service().analyze(image_bytes, triage_mode=triage_mode)
