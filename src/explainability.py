"""Task 7C: Grad-CAM Visual Interpretability & Saliency Analysis.

Provides:
- `get_target_layers`: Retrieves target convolution layers for ResNet-50 and EfficientNet-B0.
- `DermatosisExplainer`: Dual-backbone Grad-CAM explainer computing ResNet-50,
  EfficientNet-B0, and Fused Ensemble (0.5/0.5) saliency maps across facial and tile patches.
- Reconstructs full-image attention heatmaps and calculates quantitative patch attribution scores.
- CLI interface for explaining individual skin images.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_eval_transforms,
)
from src.model import build_model, DermatosisClassifier
from src.ensemble import (
    DEFAULT_RESNET_CHECKPOINT,
    DEFAULT_EFFNET_CHECKPOINT,
)
from src.detection import (
    FaceDetector,
    FaceLandmarker,
    extract_face_patches,
    extract_tile_patches,
)


def get_target_layers(classifier: Union[DermatosisClassifier, nn.Module]) -> List[nn.Module]:
    """Retrieve target feature map convolution layers for Grad-CAM.
    
    For ResNet-50: final bottleneck block (layer4[-1]), producing 7x7 spatial maps.
    For EfficientNet-B0: conv_head (final conv block before global average pooling).
    """
    backbone = classifier.model if hasattr(classifier, "model") else classifier

    if hasattr(backbone, "layer4") and len(backbone.layer4) > 0:
        return [backbone.layer4[-1]]
    elif hasattr(backbone, "conv_head"):
        return [backbone.conv_head]
    elif hasattr(backbone, "bn2"):
        return [backbone.bn2]
    else:
        # Fallback: find the last 2D convolution layer
        conv_layers = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        raise ValueError(f"Could not automatically identify target conv layer in {type(backbone)}")


class DermatosisExplainer:
    """Grad-CAM explainer for ResNet-50, EfficientNet-B0, and their 0.5/0.5 Ensemble."""

    def __init__(
        self,
        resnet_checkpoint: Union[str, Path] = DEFAULT_RESNET_CHECKPOINT,
        effnet_checkpoint: Union[str, Path] = DEFAULT_EFFNET_CHECKPOINT,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        resnet_checkpoint = Path(resnet_checkpoint)
        effnet_checkpoint = Path(effnet_checkpoint)

        if not resnet_checkpoint.exists():
            raise FileNotFoundError(f"ResNet-50 checkpoint not found at {resnet_checkpoint}")
        if not effnet_checkpoint.exists():
            raise FileNotFoundError(f"EfficientNet-B0 checkpoint not found at {effnet_checkpoint}")

        # Load ResNet-50
        self.resnet = build_model(
            model_name="resnet50",
            num_classes=NUM_CLASSES,
            pretrained=False,
            drop_rate=0.2,
            device=self.device,
        )
        r_ckpt = torch.load(resnet_checkpoint, map_location=self.device, weights_only=False)
        self.resnet.load_state_dict(r_ckpt["model_state_dict"])
        self.resnet.eval()
        self.resnet.requires_grad_(True)

        # Load EfficientNet-B0
        self.effnet = build_model(
            model_name="efficientnet_b0",
            num_classes=NUM_CLASSES,
            pretrained=False,
            drop_rate=0.2,
            device=self.device,
        )
        e_ckpt = torch.load(effnet_checkpoint, map_location=self.device, weights_only=False)
        self.effnet.load_state_dict(e_ckpt["model_state_dict"])
        self.effnet.eval()
        self.effnet.requires_grad_(True)

        # Initialize GradCAM engines
        self.cam_resnet = GradCAM(
            model=self.resnet,
            target_layers=get_target_layers(self.resnet),
        )
        self.cam_effnet = GradCAM(
            model=self.effnet,
            target_layers=get_target_layers(self.effnet),
        )

        self.eval_transforms = get_eval_transforms(img_size=224)
        self.detector: Optional[FaceDetector] = None
        self.landmarker: Optional[FaceLandmarker] = None

        logging.info(f"Initialized DermatosisExplainer on device: {self.device}")

    def _ensure_detectors(self) -> None:
        """Lazy initialization of MediaPipe detectors."""
        if self.detector is None:
            self.detector = FaceDetector()
        if self.landmarker is None:
            self.landmarker = FaceLandmarker()

    def explain_patch_batch(
        self,
        patch_tensors: torch.Tensor,
        patch_rgb_floats: List[np.ndarray],
        target_class_idx: Optional[int] = None,
        patch_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Explain a batch of K 224x224 skin patches.
        
        Args:
            patch_tensors: Tensor of shape (K, 3, 224, 224), normalized float32.
            patch_rgb_floats: List of K numpy arrays (224, 224, 3) in [0.0, 1.0] RGB.
            target_class_idx: Optional target class index (0..3). If None, defaults to
                the Ensemble's predicted class.
            patch_names: Optional human-readable region names for the K patches.
            
        Returns:
            Dict containing predictions, individual and fused CAMs, RGB overlays,
            patch attribution scores, and the dominant diagnostic patch.
        """
        K = patch_tensors.size(0)
        if patch_names is None:
            patch_names = [f"patch_{i}" for i in range(K)]

        # Move tensors to device in float32
        x = patch_tensors.to(self.device, dtype=torch.float32)

        # Compute model probabilities with autograd enabled
        with torch.enable_grad():
            r_logits = self.resnet(x)
            r_probs = F.softmax(r_logits, dim=-1)

            e_logits = self.effnet(x)
            e_probs = F.softmax(e_logits, dim=-1)

            ens_probs = 0.5 * r_probs + 0.5 * e_probs

            # Image-level mean probabilities across the K patches
            r_img_probs = r_probs.mean(dim=0).detach().cpu().numpy()
            e_img_probs = e_probs.mean(dim=0).detach().cpu().numpy()
            ens_img_probs = ens_probs.mean(dim=0).detach().cpu().numpy()

            if target_class_idx is None:
                target_class_idx = int(np.argmax(ens_img_probs))

            target_class_name = CLASS_NAMES[target_class_idx]
            targets = [ClassifierOutputTarget(target_class_idx)] * K

            # Run Grad-CAM
            resnet_cams = self.cam_resnet(input_tensor=x, targets=targets)
            effnet_cams = self.cam_effnet(input_tensor=x, targets=targets)

        # Detach probability tensors
        r_probs_np = r_probs.detach().cpu().numpy()
        e_probs_np = e_probs.detach().cpu().numpy()
        ens_probs_np = ens_probs.detach().cpu().numpy()

        # Compute Fused Ensemble CAM: 0.5 * CAM_resnet + 0.5 * CAM_effnet
        ens_cams = np.zeros_like(resnet_cams)
        for k in range(K):
            raw_fuse = 0.5 * resnet_cams[k] + 0.5 * effnet_cams[k]
            f_min = float(raw_fuse.min())
            f_max = float(raw_fuse.max())
            if f_max > f_min + 1e-6:
                ens_cams[k] = (raw_fuse - f_min) / (f_max - f_min)
            else:
                ens_cams[k] = raw_fuse

        # Generate uint8 RGB overlays
        resnet_overlays = []
        effnet_overlays = []
        ens_overlays = []

        patch_attributions = []
        for k in range(K):
            rgb_f = patch_rgb_floats[k]
            if rgb_f.shape[:2] != (224, 224):
                rgb_f = cv2.resize(rgb_f, (224, 224))
            rgb_f = np.clip(rgb_f, 0.0, 1.0).astype(np.float32)

            ov_r = show_cam_on_image(rgb_f, resnet_cams[k], use_rgb=True)
            ov_e = show_cam_on_image(rgb_f, effnet_cams[k], use_rgb=True)
            ov_ens = show_cam_on_image(rgb_f, ens_cams[k], use_rgb=True)

            resnet_overlays.append(ov_r)
            effnet_overlays.append(ov_e)
            ens_overlays.append(ov_ens)

            mean_act = float(np.mean(ens_cams[k]))
            q75_act = float(np.percentile(ens_cams[k], 75))
            max_act = float(np.max(ens_cams[k]))

            patch_attributions.append({
                "patch_index": k,
                "patch_name": patch_names[k],
                "mean_cam_activation": mean_act,
                "q75_cam_activation": q75_act,
                "max_cam_activation": max_act,
                "resnet_mean_cam": float(np.mean(resnet_cams[k])),
                "effnet_mean_cam": float(np.mean(effnet_cams[k])),
                "resnet_target_prob": float(r_probs_np[k, target_class_idx]),
                "effnet_target_prob": float(e_probs_np[k, target_class_idx]),
                "ens_target_prob": float(ens_probs_np[k, target_class_idx]),
            })

        # Identify dominant diagnostic patch (highest mean ensemble CAM activation)
        dominant_idx = int(np.argmax([attr["mean_cam_activation"] for attr in patch_attributions]))
        dominant_patch = patch_attributions[dominant_idx]

        return {
            "target_class_idx": target_class_idx,
            "target_class_name": target_class_name,
            "predictions": {
                "resnet_pred": CLASS_NAMES[int(np.argmax(r_img_probs))],
                "resnet_prob": float(r_img_probs[target_class_idx]),
                "resnet_all_probs": {cls: float(r_img_probs[i]) for i, cls in enumerate(CLASS_NAMES)},
                "effnet_pred": CLASS_NAMES[int(np.argmax(e_img_probs))],
                "effnet_prob": float(e_img_probs[target_class_idx]),
                "effnet_all_probs": {cls: float(e_img_probs[i]) for i, cls in enumerate(CLASS_NAMES)},
                "ens_pred": CLASS_NAMES[int(np.argmax(ens_img_probs))],
                "ens_prob": float(ens_img_probs[target_class_idx]),
                "ens_all_probs": {cls: float(ens_img_probs[i]) for i, cls in enumerate(CLASS_NAMES)},
            },
            "patch_probs": {
                "resnet": r_probs_np,
                "effnet": e_probs_np,
                "ensemble": ens_probs_np,
            },
            "cams": {
                "resnet": resnet_cams,
                "effnet": effnet_cams,
                "ensemble": ens_cams,
            },
            "overlays": {
                "resnet": resnet_overlays,
                "effnet": effnet_overlays,
                "ensemble": ens_overlays,
            },
            "patch_attributions": patch_attributions,
            "dominant_patch": dominant_patch,
        }

    def explain_raw_image(
        self,
        image_path: Union[str, Path],
        target_class_idx: Optional[int] = None,
        output_png_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a raw image file, extract face/tile patches, and generate saliency maps.
        
        Projects patch CAMs back onto original image coordinates to produce a reconstructed
        full-image heatmap overlay.
        """
        self._ensure_detectors()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not decode image at {image_path}")

        orig_h, orig_w = img_bgr.shape[:2]

        # Run face detection
        det_info = self.detector.detect(img_bgr)
        is_face = det_info.get("face_detected", False)

        patch_bgrs = []
        patch_names = []
        boxes = []
        canvas_bgr = img_bgr

        if is_face:
            face_res = extract_face_patches(
                img_bgr,
                detection_info=det_info,
                landmarker=self.landmarker,
                target_size=224,
            )
            for reg_name in ["forehead", "nose", "left_cheek", "right_cheek", "chin"]:
                if reg_name in face_res["patches"]:
                    patch_bgrs.append(face_res["patches"][reg_name])
                    patch_names.append(reg_name)
                    boxes.append(face_res["boxes"][reg_name])
        else:
            tile_res = extract_tile_patches(img_bgr, target_size=224, max_tiles=9)
            canvas_bgr = tile_res["image_tiled"]
            for idx, p_bgr in enumerate(tile_res["patches"]):
                patch_bgrs.append(p_bgr)
                patch_names.append(f"tile_{idx}")
                boxes.append(tile_res["boxes"][idx])

        # Prepare normalized tensors and RGB floats
        patch_tensors = []
        patch_rgb_floats = []
        for p_bgr in patch_bgrs:
            p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
            p_float = p_rgb.astype(np.float32) / 255.0
            patch_rgb_floats.append(p_float)
            p_tensor = self.eval_transforms(p_rgb)
            patch_tensors.append(p_tensor)

        stacked_tensors = torch.stack(patch_tensors, dim=0)

        # Run patch batch explanation
        result = self.explain_patch_batch(
            patch_tensors=stacked_tensors,
            patch_rgb_floats=patch_rgb_floats,
            target_class_idx=target_class_idx,
            patch_names=patch_names,
        )

        # Project patch CAMs back onto full image canvas
        cH, cW = canvas_bgr.shape[:2]
        full_cam_resnet = np.zeros((cH, cW), dtype=np.float32)
        full_cam_effnet = np.zeros((cH, cW), dtype=np.float32)
        full_cam_ens = np.zeros((cH, cW), dtype=np.float32)
        full_counts = np.zeros((cH, cW), dtype=np.float32)

        for k, (x1, y1, x2, y2) in enumerate(boxes):
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            r_cam_res = cv2.resize(result["cams"]["resnet"][k], (bw, bh), interpolation=cv2.INTER_LINEAR)
            e_cam_res = cv2.resize(result["cams"]["effnet"][k], (bw, bh), interpolation=cv2.INTER_LINEAR)
            ens_cam_res = cv2.resize(result["cams"]["ensemble"][k], (bw, bh), interpolation=cv2.INTER_LINEAR)

            full_cam_resnet[y1:y2, x1:x2] += r_cam_res
            full_cam_effnet[y1:y2, x1:x2] += e_cam_res
            full_cam_ens[y1:y2, x1:x2] += ens_cam_res
            full_counts[y1:y2, x1:x2] += 1.0

        mask = full_counts > 0
        full_cam_resnet[mask] /= full_counts[mask]
        full_cam_effnet[mask] /= full_counts[mask]
        full_cam_ens[mask] /= full_counts[mask]

        # Normalize full CAMs
        for c_arr in [full_cam_resnet, full_cam_effnet, full_cam_ens]:
            c_min = float(c_arr.min())
            c_max = float(c_arr.max())
            if c_max > c_min + 1e-6:
                c_arr[:] = (c_arr - c_min) / (c_max - c_min)

        canvas_rgb_float = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        full_overlay_ens = show_cam_on_image(canvas_rgb_float, full_cam_ens, use_rgb=True)

        result["image_info"] = {
            "image_path": str(image_path),
            "face_detected": is_face,
            "canvas_shape": (cH, cW),
            "num_patches": len(patch_bgrs),
            "boxes": boxes,
        }
        result["full_cam_ensemble"] = full_cam_ens
        result["full_overlay_ensemble"] = full_overlay_ens
        result["canvas_rgb_float"] = canvas_rgb_float

        if output_png_path is not None:
            self.save_explanation_figure(
                result=result,
                patch_rgb_floats=patch_rgb_floats,
                output_path=output_png_path,
                title=title,
            )

        return result

    def save_explanation_figure(
        self,
        result: Dict[str, Any],
        patch_rgb_floats: List[np.ndarray],
        output_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> None:
        """Create and save a high-resolution multi-panel visual explanation figure."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        K = len(patch_rgb_floats)
        preds = result["predictions"]
        tgt_name = result["target_class_name"]

        # Figure layout: Left side overview (Full Image + Full CAM), Right side patch matrix
        # Columns: [Original, ResNet CAM, EffNet CAM, Ensemble CAM]
        ncols = 4
        nrows = K
        fig = plt.figure(figsize=(16, 3.2 * max(nrows, 2)), dpi=120)

        main_title = title if title is not None else (
            f"Grad-CAM Attribution: Target = {tgt_name}\n"
            f"Ensemble: {preds['ens_pred']} ({preds['ens_prob']*100:.1f}%) | "
            f"ResNet-50: {preds['resnet_pred']} ({preds['resnet_prob']*100:.1f}%) | "
            f"EfficientNet-B0: {preds['effnet_pred']} ({preds['effnet_prob']*100:.1f}%)"
        )
        fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.99)

        col_titles = [
            "Original Patch",
            f"ResNet-50 CAM ({tgt_name})",
            f"EfficientNet-B0 CAM ({tgt_name})",
            f"Ensemble Fused CAM ({tgt_name})",
        ]

        for k in range(K):
            attr = result["patch_attributions"][k]
            p_name = attr["patch_name"]
            is_dom = (k == result["dominant_patch"]["patch_index"])
            dom_flag = " [DOMINANT]" if is_dom else ""

            # 1. Original patch
            ax0 = plt.subplot(nrows, ncols, k * ncols + 1)
            ax0.imshow(patch_rgb_floats[k])
            ax0.set_title(f"{p_name}{dom_flag}", fontsize=10, fontweight="bold" if is_dom else "normal")
            ax0.axis("off")
            if k == 0:
                ax0.annotate(col_titles[0], xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold")

            # 2. ResNet CAM
            ax1 = plt.subplot(nrows, ncols, k * ncols + 2)
            ax1.imshow(result["overlays"]["resnet"][k])
            r_p = attr["resnet_target_prob"] * 100
            ax1.set_title(f"P={r_p:.1f}% | act={attr['resnet_mean_cam']:.2f}", fontsize=9)
            ax1.axis("off")
            if k == 0:
                ax1.annotate(col_titles[1], xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold", color="#1f77b4")

            # 3. EffNet CAM
            ax2 = plt.subplot(nrows, ncols, k * ncols + 3)
            ax2.imshow(result["overlays"]["effnet"][k])
            e_p = attr["effnet_target_prob"] * 100
            ax2.set_title(f"P={e_p:.1f}% | act={attr['effnet_mean_cam']:.2f}", fontsize=9)
            ax2.axis("off")
            if k == 0:
                ax2.annotate(col_titles[2], xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold", color="#ff7f0e")

            # 4. Ensemble CAM
            ax3 = plt.subplot(nrows, ncols, k * ncols + 4)
            ax3.imshow(result["overlays"]["ensemble"][k])
            ens_p = attr["ens_target_prob"] * 100
            ax3.set_title(f"P={ens_p:.1f}% | act={attr['mean_cam_activation']:.2f}", fontsize=9, fontweight="bold" if is_dom else "normal")
            ax3.axis("off")
            if k == 0:
                ax3.annotate(col_titles[3], xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold", color="#2ca02c")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved explanation figure to {output_path}")

    def close(self) -> None:
        """Release GradCAM hooks, detector handles, and free resources."""
        if hasattr(self, "cam_resnet") and self.cam_resnet is not None:
            if hasattr(self.cam_resnet, "activations_and_grads"):
                self.cam_resnet.activations_and_grads.release()
            self.cam_resnet = None
        if hasattr(self, "cam_effnet") and self.cam_effnet is not None:
            if hasattr(self.cam_effnet, "activations_and_grads"):
                self.cam_effnet.activations_and_grads.release()
            self.cam_effnet = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self) -> DermatosisExplainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grad-CAM visual explanation on skin image.")
    parser.add_argument("--image", type=str, required=True, help="Path to raw dermatology image")
    parser.add_argument("--output", type=str, default="gradcam_output.png", help="Path to save output visualization")
    parser.add_argument("--target-class", type=str, default=None, choices=CLASS_NAMES, help="Optional target class override")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    target_idx = CLASS_TO_IDX[args.target_class] if args.target_class else None

    with DermatosisExplainer(device=args.device) as explainer:
        res = explainer.explain_raw_image(
            image_path=args.image,
            target_class_idx=target_idx,
            output_png_path=args.output,
        )
        print("\n=== Grad-CAM Explanation Result ===")
        print(f"Target Class: {res['target_class_name']}")
        print(f"Ensemble: {res['predictions']['ens_pred']} ({res['predictions']['ens_prob']*100:.2f}%)")
        print(f"ResNet-50: {res['predictions']['resnet_pred']} ({res['predictions']['resnet_prob']*100:.2f}%)")
        print(f"EfficientNet-B0: {res['predictions']['effnet_pred']} ({res['predictions']['effnet_prob']*100:.2f}%)")
        print(f"Dominant Diagnostic Patch: {res['dominant_patch']['patch_name']} (mean act = {res['dominant_patch']['mean_cam_activation']:.3f})")
        print(f"Visualization saved to: {args.output}\n")


if __name__ == "__main__":
    main()
