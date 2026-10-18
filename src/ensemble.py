"""ResNet-50 + EfficientNet-B0 Model Ensemble.

Provides:
- `DermatosisEnsemble`: Soft probability weighted fusion of ResNet-50 and EfficientNet-B0.
- `find_optimal_ensemble_weights`: Grid search on validation split to find optimal fusion weights.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX
from src.model import build_model, DermatosisClassifier

DEFAULT_RESNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet50_baseline_best.pt"
DEFAULT_EFFNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_baseline_best.pt"


class DermatosisEnsemble(nn.Module):
    """Dual-backbone model ensemble fusing ResNet-50 and EfficientNet-B0 softmax probabilities."""

    def __init__(
        self,
        resnet_checkpoint: Union[str, Path] = DEFAULT_RESNET_CHECKPOINT,
        effnet_checkpoint: Union[str, Path] = DEFAULT_EFFNET_CHECKPOINT,
        weights: Tuple[float, float] = (0.5, 0.5),
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

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

        # Instantiate backbones
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

        self.set_weights(weights)
        logging.info(
            f"Initialized DermatosisEnsemble on {self.device} with weights: "
            f"ResNet-50={self.w_resnet:.2f}, EfficientNet-B0={self.w_effnet:.2f}"
        )

    def set_weights(self, weights: Tuple[float, float]) -> None:
        """Set and normalize ensemble fusion weights (w_resnet, w_effnet)."""
        w_res, w_eff = float(weights[0]), float(weights[1])
        total = w_res + w_eff
        if total <= 0:
            raise ValueError(f"Ensemble weights must sum to > 0, got {weights}")
        self.w_resnet = w_res / total
        self.w_effnet = w_eff / total
        self.weights = (self.w_resnet, self.w_effnet)

    @torch.no_grad()
    def predict_patches(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs mixed-precision forward pass on patch batch x through both backbones.
        
        Args:
            x: Tensor of shape (K, 3, 224, 224)
            
        Returns:
            Dict containing:
            - 'ensemble_probs': weighted patch probabilities (K, num_classes)
            - 'resnet_probs': ResNet-50 patch probabilities (K, num_classes)
            - 'effnet_probs': EfficientNet-B0 patch probabilities (K, num_classes)
        """
        x = x.to(self.device, non_blocking=True)
        device_type = self.device.type if isinstance(self.device, torch.device) else "cuda"

        with torch.amp.autocast(device_type):
            resnet_logits = self.resnet(x)
            effnet_logits = self.effnet(x)

            resnet_probs = F.softmax(resnet_logits, dim=-1)
            effnet_probs = F.softmax(effnet_logits, dim=-1)

            ensemble_probs = self.w_resnet * resnet_probs + self.w_effnet * effnet_probs

        return {
            "ensemble_probs": ensemble_probs,
            "resnet_probs": resnet_probs,
            "effnet_probs": effnet_probs,
        }

    @torch.no_grad()
    def predict_image_from_patches(
        self,
        x: torch.Tensor,
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """Aggregate patch probabilities into image-level predictions for each backbone and ensemble.
        
        Args:
            x: Tensor of shape (K, 3, 224, 224)
            aggregation: 'mean' or 'max' pooling across patches.
            
        Returns:
            Dict containing ensemble and per-backbone image-level probabilities, predicted classes,
            and patch-level probabilities.
        """
        patch_res = self.predict_patches(x)
        r_probs = patch_res["resnet_probs"]
        e_probs = patch_res["effnet_probs"]

        if aggregation == "mean":
            r_img_probs = r_probs.mean(dim=0)
            e_img_probs = e_probs.mean(dim=0)
        elif aggregation == "max":
            r_img_probs = r_probs.max(dim=0).values
            r_img_probs = r_img_probs / r_img_probs.sum()
            e_img_probs = e_probs.max(dim=0).values
            e_img_probs = e_img_probs / e_img_probs.sum()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        ens_img_probs = self.w_resnet * r_img_probs + self.w_effnet * e_img_probs

        pred_idx = int(torch.argmax(ens_img_probs).item())
        resnet_pred_idx = int(torch.argmax(r_img_probs).item())
        effnet_pred_idx = int(torch.argmax(e_img_probs).item())

        return {
            "ensemble_probs": ens_img_probs,
            "resnet_probs": r_img_probs,
            "effnet_probs": e_img_probs,
            "pred_idx": pred_idx,
            "predicted_class": IDX_TO_CLASS[pred_idx],
            "resnet_pred_idx": resnet_pred_idx,
            "resnet_predicted_class": IDX_TO_CLASS[resnet_pred_idx],
            "effnet_pred_idx": effnet_pred_idx,
            "effnet_predicted_class": IDX_TO_CLASS[effnet_pred_idx],
            "models_agree": (resnet_pred_idx == effnet_pred_idx),
            "ensemble_patch_probs": patch_res["ensemble_probs"],
            "resnet_patch_probs": r_probs,
            "effnet_patch_probs": e_probs,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass returns weighted ensemble patch probabilities."""
        return self.predict_patches(x)["ensemble_probs"]


def find_optimal_ensemble_weights(
    resnet_img_probs: np.ndarray,
    effnet_img_probs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """1D grid search over w_resnet in [0.00, 1.00] to maximize Macro-F1 on validation data.
    
    Tie-breaking hierarchy:
    1. Highest Macro-F1
    2. Highest Accuracy
    3. Closeness to 0.50 (equal weighting)
    
    Args:
        resnet_img_probs: Array of shape (N, num_classes) containing ResNet-50 image probabilities.
        effnet_img_probs: Array of shape (N, num_classes) containing EfficientNet-B0 image probabilities.
        targets: Array of shape (N,) containing ground truth class indices.
        
    Returns:
        Dict containing best_w_resnet, best_w_effnet, best_macro_f1, best_accuracy, and grid_results list.
    """
    grid_results = []
    best_macro_f1 = -1.0
    best_acc = -1.0
    best_w_resnet = 0.50
    min_diff_from_half = 1.0

    w_values = np.round(np.arange(0.0, 1.001, 0.05), 2)
    for w_res in w_values:
        w_res = float(w_res)
        w_eff = round(float(1.0 - w_res), 2)

        fused_probs = w_res * resnet_img_probs + w_eff * effnet_img_probs
        preds = np.argmax(fused_probs, axis=-1)

        f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(targets, preds))

        grid_results.append({
            "w_resnet": round(w_res, 2),
            "w_effnet": round(w_eff, 2),
            "val_macro_f1": float(f1),
            "val_accuracy": float(acc),
        })

        diff_from_half = abs(w_res - 0.5)
        is_better = False
        if f1 > best_macro_f1 + 1e-6:
            is_better = True
        elif abs(f1 - best_macro_f1) <= 1e-6:
            if acc > best_acc + 1e-6:
                is_better = True
            elif abs(acc - best_acc) <= 1e-6:
                if diff_from_half < min_diff_from_half - 1e-6:
                    is_better = True

        if is_better:
            best_macro_f1 = f1
            best_acc = acc
            best_w_resnet = round(w_res, 2)
            min_diff_from_half = diff_from_half

    best_w_effnet = round(float(1.0 - best_w_resnet), 2)

    logging.info(
        f"Optimal ensemble weights on validation set: "
        f"ResNet-50={best_w_resnet:.2f}, EfficientNet-B0={best_w_effnet:.2f} "
        f"(Val Macro-F1={best_macro_f1:.4f}, Val Acc={best_acc * 100:.2f}%)"
    )

    return {
        "best_w_resnet": best_w_resnet,
        "best_w_effnet": best_w_effnet,
        "best_macro_f1": best_macro_f1,
        "best_accuracy": best_acc,
        "grid_results": grid_results,
    }
