"""PyTorch Model Architecture, Loss Functions, and Patch Aggregation.

Provides:
- `DermatosisClassifier`: Backbone wrapper using timm (ResNet-50 baseline and future backbones).
- Inverse-frequency weighted CrossEntropyLoss.
- Patch aggregation utilities (mean-probability pooling across face & close-up patches).
- Checkpoint saving and loading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import CLASS_NAMES, NUM_CLASSES


class DermatosisClassifier(nn.Module):
    """Deep learning classifier for facial dermatosis patch classification."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Instantiate backbone via timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns unnormalized logits of shape (B, num_classes)."""
        return self.model(x)

    @torch.no_grad()
    def predict_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape (B, num_classes)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def aggregate_patches(
        patch_probs: torch.Tensor,
        method: str = "mean",
    ) -> Tuple[torch.Tensor, int]:
        """Aggregate patch probabilities for a single image into an image-level prediction.
        
        Args:
            patch_probs: Tensor of shape (K, num_classes) where K is number of patches.
            method: 'mean' (softmax probability pooling) or 'max'.
            
        Returns:
            image_probs: Tensor of shape (num_classes,)
            pred_idx: integer index of the predicted class
        """
        if method == "mean":
            img_probs = patch_probs.mean(dim=0)
        elif method == "max":
            img_probs = patch_probs.max(dim=0).values
            img_probs = img_probs / img_probs.sum()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        pred_idx = int(torch.argmax(img_probs).item())
        return img_probs, pred_idx


def build_model(
    model_name: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
) -> DermatosisClassifier:
    """Factory function to instantiate and place model onto device."""
    model = DermatosisClassifier(
        backbone_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
    if device is not None:
        model = model.to(device)
    return model


def get_loss_function(class_weights: Optional[torch.Tensor] = None) -> nn.CrossEntropyLoss:
    """Return standard or class-weighted Cross-Entropy Loss."""
    return nn.CrossEntropyLoss(weight=class_weights)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: Path,
    is_best: bool = False,
):
    """Save model checkpoint safely."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = checkpoint_path.parent / (checkpoint_path.stem + "_best.pt")
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load model checkpoint and return checkpoint dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint
