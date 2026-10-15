"""End-to-end inference pipeline for facial dermatosis diagnosis.

Given an unseen smartphone image (selfie or close-up):
1. Runs MediaPipe FaceDetector to detect faces.
2. Extracts landmarked/proportional anatomical patches (forehead, nose, cheeks, chin) for selfies,
   or tiles for close-ups.
3. Classifies each patch using the trained DermatosisClassifier (ResNet-50).
4. Aggregates patch probabilities (mean pooling) into a calibrated diagnosis.
5. Returns diagnosis, confidence, class probabilities, and per-patch breakdown.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_eval_transforms,
)
from src.detection import (
    FaceDetector,
    FaceLandmarker,
    extract_face_patches,
    extract_tile_patches,
)
from src.model import build_model, load_checkpoint, DermatosisClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet50_baseline_best.pt"


class FacialDermatosisPredictor:
    """Production inference wrapper combining face detection, patch cropping, and classification."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        backbone_name: str = "resnet50",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = build_model(
            model_name=backbone_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
            device=self.device,
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        load_checkpoint(checkpoint_path, self.model, device=str(self.device))
        self.model.eval()

        self.transform = get_eval_transforms()
        self.detector = FaceDetector()
        self.landmarker = FaceLandmarker()

    def predict(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """Predict dermatosis from an arbitrary smartphone photo.
        
        Args:
            image_or_path: File path or BGR numpy array.
            aggregation: 'mean' or 'max' patch probability pooling.
            
        Returns:
            Dict containing diagnosis, confidence, class probabilities, and patch details.
        """
        if isinstance(image_or_path, (str, Path)):
            p = str(image_or_path)
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                raise ValueError(f"Could not read image at {p}")
        else:
            img_bgr = image_or_path

        h, w = img_bgr.shape[:2]
        det = self.detector.detect(img_bgr)
        is_face = bool(det.get("face_detected", False))

        patch_items = []
        if is_face:
            try:
                face_res = extract_face_patches(img_bgr, det, self.landmarker, target_size=224)
                for reg_name, p_bgr in face_res["patches"].items():
                    patch_items.append((reg_name, p_bgr, face_res["boxes"].get(reg_name)))
            except Exception:
                tile_res = extract_tile_patches(img_bgr, target_size=224, max_tiles=9)
                for i, p_bgr in enumerate(tile_res["patches"]):
                    patch_items.append((f"tile_{i}", p_bgr, tile_res["boxes"][i]))
        else:
            tile_res = extract_tile_patches(img_bgr, target_size=224, max_tiles=9)
            for i, p_bgr in enumerate(tile_res["patches"]):
                patch_items.append((f"tile_{i}", p_bgr, tile_res["boxes"][i]))

        if not patch_items:
            raise RuntimeError("Failed to extract any patches from the image.")

        # Prepare batch of patches
        tensors = []
        for _, p_bgr, _ in patch_items:
            p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(p_rgb))

        stacked = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                logits = self.model(stacked)
                patch_probs = torch.softmax(logits, dim=-1)

        # Image-level aggregation
        img_probs, pred_idx = self.model.aggregate_patches(patch_probs, method=aggregation)
        predicted_class = IDX_TO_CLASS[pred_idx]
        confidence = float(img_probs[pred_idx].item())

        class_probabilities = {
            cls_name: round(float(img_probs[c_idx].item()), 4)
            for c_idx, cls_name in enumerate(CLASS_NAMES)
        }

        # Per-patch breakdown
        patch_breakdown = []
        for idx, (p_name, _, bbox) in enumerate(patch_items):
            p_p = patch_probs[idx].cpu().numpy()
            p_pred = IDX_TO_CLASS[int(np.argmax(p_p))]
            patch_breakdown.append({
                "patch_name": p_name,
                "bbox": bbox,
                "predicted_class": p_pred,
                "confidence": round(float(np.max(p_p)), 4),
                "probabilities": {cls_name: round(float(p_p[c_idx]), 4) for c_idx, cls_name in enumerate(CLASS_NAMES)},
            })

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "face_detected": is_face,
            "num_patches": len(patch_items),
            "patch_breakdown": patch_breakdown,
        }

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Predict dermatosis condition on an image")
    parser.add_argument("--image", type=Path, required=True, help="Path to input photo")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    args = parser.parse_args()

    with FacialDermatosisPredictor(checkpoint_path=args.checkpoint) as predictor:
        res = predictor.predict(args.image)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Predicted Class: {res['predicted_class']} (Confidence: {res['confidence'] * 100:.2f}%)")
        print(f"Face Detected:   {res['face_detected']} ({res['num_patches']} patches evaluated)")
        print("\nClass Probabilities:")
        for c, p in res["class_probabilities"].items():
            print(f"  {c:12s}: {p * 100:.2f}%")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
