import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"

MODEL_URLS = {
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def ensure_model(model_name: str) -> Path:
    """Download MediaPipe model from Google's public storage if not present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / model_name
    if not target.exists():
        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_URLS.keys())}")
        url = MODEL_URLS[model_name]
        urllib.request.urlretrieve(url, target)
    return target


class FaceDetector:
    """MediaPipe Tasks FaceDetector wrapper for dermatology selfies and close-ups."""

    def __init__(self, model_path: Optional[Path] = None, min_detection_confidence: float = 0.5):
        if model_path is None:
            model_path = ensure_model("blaze_face_short_range.tflite")
        self.model_path = Path(model_path)
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image_or_path: Any, max_dim: int = 1280) -> Dict[str, Any]:
        """Run face detection on an image.
        
        Args:
            image_or_path: BGR numpy image or file path.
            max_dim: Max length of the long edge for detection.
            
        Returns:
            dict containing face_detected (bool), face_conf (float or None),
            face_area_frac (float or None), bbox (x, y, w, h in original pixels),
            keypoints (list of (x, y) tuples in original pixels).
        """
        if isinstance(image_or_path, (str, Path)):
            path_str = str(image_or_path)
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Could not read image at {path_str}")
        else:
            img = image_or_path

        orig_h, orig_w = img.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return {
                "face_detected": False,
                "face_conf": None,
                "face_area_frac": None,
                "bbox": None,
                "keypoints": None,
            }

        # Downscale for detection if long side > max_dim
        scale = 1.0
        max_edge = max(orig_h, orig_w)
        if max_edge > max_dim:
            scale = max_dim / float(max_edge)
            det_w = max(1, int(round(orig_w * scale)))
            det_h = max(1, int(round(orig_h * scale)))
            det_img = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = orig_w, orig_h
            det_img = img

        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if not result.detections:
            return {
                "face_detected": False,
                "face_conf": None,
                "face_area_frac": None,
                "bbox": None,
                "keypoints": None,
            }

        # Choose the detection with the largest bounding box area
        best_det = None
        best_area = -1.0
        for det in result.detections:
            area = det.bounding_box.width * det.bounding_box.height
            if area > best_area:
                best_area = area
                best_det = det

        bbox = best_det.bounding_box
        conf = float(best_det.categories[0].score) if best_det.categories else 1.0

        # Fractional area relative to the image
        img_area = float(det_w * det_h)
        face_area_frac = float(best_area) / img_area if img_area > 0 else 0.0

        # Map bounding box back to original coordinates
        orig_bx = int(round(bbox.origin_x / scale))
        orig_by = int(round(bbox.origin_y / scale))
        orig_bw = int(round(bbox.width / scale))
        orig_bh = int(round(bbox.height / scale))

        # Clamp to original image bounds
        orig_bx = max(0, min(orig_bx, orig_w - 1))
        orig_by = max(0, min(orig_by, orig_h - 1))
        orig_bw = max(1, min(orig_bw, orig_w - orig_bx))
        orig_bh = max(1, min(orig_bh, orig_h - orig_by))

        # Map keypoints if available (normalized coordinates [0, 1] relative to det_img)
        kps = []
        if best_det.keypoints:
            for kp in best_det.keypoints:
                kps.append((float(kp.x * orig_w), float(kp.y * orig_h)))

        return {
            "face_detected": True,
            "face_conf": conf,
            "face_area_frac": face_area_frac,
            "bbox": (orig_bx, orig_by, orig_bw, orig_bh),
            "keypoints": kps,
        }

    def close(self):
        """Release MediaPipe detector resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FaceLandmarker:
    """MediaPipe Tasks FaceLandmarker wrapper for precise facial feature localization."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = ensure_model("face_landmarker.task")
        self.model_path = Path(model_path)
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, image_bgr: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """Return 478 face mesh landmarks in normalized coords (x, y, z) or None."""
        orig_h, orig_w = image_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return None

        # FaceLandmarker works well up to ~1280px
        max_edge = max(orig_h, orig_w)
        if max_edge > 1280:
            scale = 1280.0 / float(max_edge)
            det_img = cv2.resize(image_bgr, (int(round(orig_w * scale)), int(round(orig_h * scale))), interpolation=cv2.INTER_AREA)
        else:
            det_img = image_bgr

        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None

        # Return first face landmarks (normalized [0, 1])
        return [(pt.x, pt.y, pt.z) for pt in result.face_landmarks[0]]

    def close(self):
        """Release MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_face_patches(
    image_bgr: np.ndarray,
    detection_info: Optional[Dict[str, Any]] = None,
    landmarker: Optional[FaceLandmarker] = None,
    target_size: int = 224,
) -> Dict[str, Any]:
    """Extract forehead, nose, left cheek, right cheek, and chin patches from a face.
    
    Uses 478 facial landmarks if available; falls back gracefully to face bounding box proportions.
    
    Returns:
        dict containing:
            'patches': dict of region_name -> np.ndarray (target_size, target_size, 3)
            'boxes': dict of region_name -> (x1, y1, x2, y2) in image_bgr pixels
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # If detection info not provided, create a temporary detector
    if detection_info is None:
        with FaceDetector() as detector:
            detection_info = detector.detect(image_bgr)

    if not detection_info.get("face_detected", False) or detection_info.get("bbox") is None:
        raise ValueError("Cannot extract face patches: No face detected in image.")

    bx, by, bw, bh = detection_info["bbox"]
    crop_size = max(32, int(round(bw * 0.28)))

    # Attempt landmark localization
    landmarks = None
    if landmarker is not None:
        landmarks = landmarker.detect_landmarks(image_bgr)

    if landmarks is not None and len(landmarks) >= 468:
        # Canonical MediaPipe mesh landmark indices:
        # 151: Mid forehead
        # 1: Nose tip
        # 425: Left cheek (patient's left / image right)
        # 205: Right cheek (patient's right / image left)
        # 199: Chin cup
        centers = {
            "forehead": (int(round(landmarks[151][0] * orig_w)), int(round(landmarks[151][1] * orig_h))),
            "nose": (int(round(landmarks[1][0] * orig_w)), int(round(landmarks[1][1] * orig_h))),
            "left_cheek": (int(round(landmarks[425][0] * orig_w)), int(round(landmarks[425][1] * orig_h))),
            "right_cheek": (int(round(landmarks[205][0] * orig_w)), int(round(landmarks[205][1] * orig_h))),
            "chin": (int(round(landmarks[199][0] * orig_w)), int(round(landmarks[199][1] * orig_h))),
        }
    else:
        # Proportional fallback within bounding box
        centers = {
            "forehead": (int(round(bx + 0.50 * bw)), int(round(by + 0.22 * bh))),
            "nose": (int(round(bx + 0.50 * bw)), int(round(by + 0.52 * bh))),
            "left_cheek": (int(round(bx + 0.72 * bw)), int(round(by + 0.58 * bh))),
            "right_cheek": (int(round(bx + 0.28 * bw)), int(round(by + 0.58 * bh))),
            "chin": (int(round(bx + 0.50 * bw)), int(round(by + 0.82 * bh))),
        }

    half = crop_size // 2
    patches = {}
    boxes = {}

    for region, (cx, cy) in centers.items():
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(orig_w, cx + half)
        y2 = min(orig_h, cy + half)

        # If clamped region is valid
        if x2 > x1 and y2 > y1:
            crop = image_bgr[y1:y2, x1:x2]
            patch = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        else:
            patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        patches[region] = patch
        boxes[region] = (x1, y1, x2, y2)

    return {"patches": patches, "boxes": boxes}


def extract_tile_patches(
    image_bgr: np.ndarray,
    target_size: int = 224,
    max_tiles: int = 9,
) -> Dict[str, Any]:
    """Extract non-overlapping or lightly overlapping tiles from a close-up image.
    
    Resizes image so the short side is >= target_size, capped to a max dimension to avoid
    excessive tiling, and produces up to max_tiles patches.
    
    Returns:
        dict containing:
            'patches': list of np.ndarray (target_size, target_size, 3)
            'boxes': list of (x1, y1, x2, y2) in the tiled image space
            'image_tiled': the resized image from which tiles were extracted
    """
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return {"patches": [], "boxes": [], "image_tiled": image_bgr}

    # Ensure short side >= target_size
    img_work = image_bgr
    if min(orig_h, orig_w) < target_size:
        scale = float(target_size) / float(min(orig_h, orig_w))
        new_w = max(target_size, int(round(orig_w * scale)))
        new_h = max(target_size, int(round(orig_h * scale)))
        img_work = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif max(orig_h, orig_w) > 896:
        # Scale down large close-ups so short side is still >= target_size
        scale = 896.0 / float(max(orig_h, orig_w))
        if int(round(min(orig_h, orig_w) * scale)) >= target_size:
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            img_work = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = float(target_size) / float(min(orig_h, orig_w))
            new_w = max(target_size, int(round(orig_w * scale)))
            new_h = max(target_size, int(round(orig_h * scale)))
            img_work = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H, W = img_work.shape[:2]
    n_x = max(1, int(np.ceil(W / float(target_size))))
    n_y = max(1, int(np.ceil(H / float(target_size))))

    # Reduce grid size if exceeds max_tiles
    while n_x * n_y > max_tiles:
        if n_x >= n_y and n_x > 1:
            n_x -= 1
        elif n_y > 1:
            n_y -= 1
        else:
            break

    xs = np.linspace(0, W - target_size, n_x, dtype=int) if n_x > 1 else [0]
    ys = np.linspace(0, H - target_size, n_y, dtype=int) if n_y > 1 else [0]

    patches = []
    boxes = []
    for y in ys:
        for x in xs:
            tile = img_work[y : y + target_size, x : x + target_size]
            patches.append(tile)
            boxes.append((int(x), int(y), int(x + target_size), int(y + target_size)))

    return {"patches": patches, "boxes": boxes, "image_tiled": img_work}
