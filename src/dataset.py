"""PyTorch Dataset & DataLoaders for Facial Dermatosis Patch Classification.

Supports:
1. `FacialDermatosisPatchDataset`: Iterates over individual 224x224 patches for high-throughput
   training and patch-level validation with PyTorch DataLoader workers.
2. `FacialDermatosisImageDataset`: Groups patches by original patient image for aggregate
   evaluation (e.g. mean-pooled patch predictions per image) on in-domain test and SCIN external test.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "patches_manifest.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "manifest.csv"

CLASS_NAMES: List[str] = ["Acne", "Rosacea", "Dermatitis", "Normal"]
CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Standard augmentations for dermatology patch training.
    
    Includes horizontal flipping, subtle color jitter (lighting variation across phone cameras),
    and small rotations (+/- 10 degrees).
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation and testing."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FacialDermatosisPatchDataset(Dataset):
    """PyTorch Dataset yielding individual 224x224 skin patches."""

    def __init__(
        self,
        split: str,
        patches_manifest_path: Path = PATCHES_MANIFEST_PATH,
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.transform = transform if transform is not None else get_eval_transforms()

        if not patches_manifest_path.exists():
            raise FileNotFoundError(
                f"Patches manifest not found at {patches_manifest_path}. "
                "Run `python scripts/extract_patches.py` first."
            )

        df = pd.read_csv(patches_manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)

        if len(self.data) == 0:
            logging.warning(f"No patches found for split: '{split}' in {patches_manifest_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.data.iloc[idx]
        patch_path = PROJECT_ROOT / row["patch_path"]

        # Read RGB image
        img_bgr = cv2.imread(str(patch_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not read patch image at {patch_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img_rgb)
        label_idx = CLASS_TO_IDX[row["label"]]

        meta = {
            "patch_id": str(row["patch_id"]),
            "image_id": str(row["image_id"]),
            "subject_id": str(row["subject_id"]),
            "patch_type": str(row["patch_type"]),
            "is_face_crop": bool(row["is_face_crop"]),
            "label_name": str(row["label"]),
        }

        return tensor, label_idx, meta


class FacialDermatosisImageDataset(Dataset):
    """PyTorch Dataset yielding all patches for an original image for aggregate inference."""

    def __init__(
        self,
        split: str,
        patches_manifest_path: Path = PATCHES_MANIFEST_PATH,
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.transform = transform if transform is not None else get_eval_transforms()

        df = pd.read_csv(patches_manifest_path)
        self.split_df = df[df["split"] == split].reset_index(drop=True)

        # Pre-group patch records by image_id into dict of lists for fast access
        self.image_to_patches: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.split_df.to_dict(orient="records"):
            self.image_to_patches.setdefault(r["image_id"], []).append(r)
        self.unique_image_ids = list(self.image_to_patches.keys())

    def __len__(self) -> int:
        return len(self.unique_image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        img_id = self.unique_image_ids[idx]
        patch_records = self.image_to_patches[img_id]

        patch_tensors = []
        patch_types = []
        for r in patch_records:
            patch_path = PROJECT_ROOT / r["patch_path"]
            img_bgr = cv2.imread(str(patch_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            patch_tensors.append(self.transform(img_rgb))
            patch_types.append(str(r["patch_type"]))

        if not patch_tensors:
            raise RuntimeError(f"No valid patches found for image_id {img_id}")

        # Stack patches into shape (K, 3, 224, 224)
        stacked_patches = torch.stack(patch_tensors, dim=0)

        first_r = patch_records[0]
        label_idx = CLASS_TO_IDX[first_r["label"]]

        meta = {
            "image_id": img_id,
            "subject_id": str(first_r["subject_id"]),
            "source": str(first_r["source"]),
            "label_name": str(first_r["label"]),
            "num_patches": len(patch_tensors),
            "patch_types": patch_types,
        }

        return stacked_patches, label_idx, meta


def compute_class_weights(
    split: str = "train",
    patches_manifest_path: Path = PATCHES_MANIFEST_PATH,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for Cross-Entropy Loss:
    
        w_c = N_total / (C * N_c)
    """
    df = pd.read_csv(patches_manifest_path)
    split_df = df[df["split"] == split]

    counts = split_df["label"].value_counts()
    n_total = len(split_df)
    n_classes = NUM_CLASSES

    weights = []
    for cls_name in CLASS_NAMES:
        n_c = counts.get(cls_name, 1)
        w = n_total / float(n_classes * n_c)
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        w_tensor = w_tensor.to(device)

    logging.info(f"Class counts ({split}): {dict(counts)}")
    logging.info(f"Computed class weights: {dict(zip(CLASS_NAMES, [round(w.item(), 4) for w in w_tensor]))}")
    return w_tensor


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    patches_manifest_path: Path = PATCHES_MANIFEST_PATH,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, val, test, and external_test PyTorch DataLoaders."""
    train_ds = FacialDermatosisPatchDataset(
        split="train",
        patches_manifest_path=patches_manifest_path,
        transform=get_train_transforms(),
    )
    val_ds = FacialDermatosisPatchDataset(
        split="val",
        patches_manifest_path=patches_manifest_path,
        transform=get_eval_transforms(),
    )
    test_ds = FacialDermatosisPatchDataset(
        split="test",
        patches_manifest_path=patches_manifest_path,
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # External test loader
    try:
        ext_ds = FacialDermatosisPatchDataset(
            split="external_test",
            patches_manifest_path=patches_manifest_path,
            transform=get_eval_transforms(),
        )
        ext_loader = DataLoader(
            ext_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    except Exception:
        ext_loader = None

    return train_loader, val_loader, test_loader, ext_loader
