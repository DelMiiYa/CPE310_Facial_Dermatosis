"""Step 9: PyTorch Baseline Model Training (ResNet-50).

Trains a patch-based ResNet-50 classifier on real smartphone facial images with:
1. Mixed precision (torch.amp.autocast('cuda') + GradScaler) for low-VRAM RTX 3050 (4GB).
2. Class-weighted Cross-Entropy loss ($w_c = N_{total} / (C * N_c)$) to address Dermatitis imbalance.
3. AdamW optimizer + CosineAnnealingLR.
4. Image-level aggregation (mean probability pooling) for primary validation metric (Val Image Macro-F1).
5. Comprehensive evaluation on In-Domain Test and strictly held-out SCIN External Test.
6. Generates markdown report and confusion matrices.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    FacialDermatosisPatchDataset,
    FacialDermatosisImageDataset,
    compute_class_weights,
    get_train_transforms,
    get_eval_transforms,
)
from src.model import (
    build_model,
    get_loss_function,
    save_checkpoint,
    load_checkpoint,
    DermatosisClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch using mixed precision."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_batches = len(train_loader)

    for b_idx, (inputs, targets, _) in enumerate(train_loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        if b_idx % 50 == 0 or b_idx == total_batches:
            running_loss = total_loss / len(all_preds)
            logging.info(f"   Batch [{b_idx:03d}/{total_batches:03d}] - Running Loss: {running_loss:.4f}")

    avg_loss = total_loss / len(all_preds)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_patches(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on individual patches."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets, _ in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(all_preds)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_image_level(
    model: DermatosisClassifier,
    image_dataset: FacialDermatosisImageDataset,
    device: torch.device,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Evaluate aggregate image-level performance across multiple patches."""
    model.eval()
    all_preds = []
    all_targets = []
    records = []

    for idx in range(len(image_dataset)):
        stacked_patches, target_idx, meta = image_dataset[idx]
        stacked_patches = stacked_patches.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(stacked_patches)
            patch_probs = torch.softmax(logits, dim=-1)

        img_probs, pred_idx = model.aggregate_patches(patch_probs, method=aggregation)

        all_preds.append(pred_idx)
        all_targets.append(target_idx)

        record = {
            "image_id": meta["image_id"],
            "subject_id": meta["subject_id"],
            "source": meta["source"],
            "true_label": meta["label_name"],
            "pred_label": IDX_TO_CLASS[pred_idx],
            "correct": (pred_idx == target_idx),
            "num_patches": meta["num_patches"],
        }
        for c_idx, c_name in enumerate(CLASS_NAMES):
            record[f"prob_{c_name}"] = float(img_probs[c_idx].item())
        records.append(record)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    cls_report = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES, digits=4, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, [float(f) for f in per_class_f1])),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "predictions_df": pd.DataFrame(records),
    }


def train_baseline(
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    patience: int = 6,
) -> Dict[str, Any]:
    """Train ResNet-50 baseline on single seed."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Training ResNet-50 Baseline (Seed {seed}) on {device} ---")

    # Datasets
    train_patch_ds = FacialDermatosisPatchDataset(split="train", transform=get_train_transforms())
    val_patch_ds = FacialDermatosisPatchDataset(split="val", transform=get_eval_transforms())
    val_image_ds = FacialDermatosisImageDataset(split="val", transform=get_eval_transforms())
    test_image_ds = FacialDermatosisImageDataset(split="test", transform=get_eval_transforms())

    ext_image_ds = None
    try:
        ext_image_ds = FacialDermatosisImageDataset(split="external_test", transform=get_eval_transforms())
    except Exception as e:
        logging.warning(f"Could not load external_test: {e}")

    logging.info(f"Loaded {len(train_patch_ds)} train patches, {len(val_patch_ds)} val patches.")
    logging.info(f"Image dataset: {len(val_image_ds)} val images, {len(test_image_ds)} test images.")

    # DataLoaders
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_patch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Class Weights and Loss
    class_weights = compute_class_weights(split="train", device=device)
    criterion = get_loss_function(class_weights=class_weights)

    # Model
    model = build_model(model_name="resnet50", num_classes=NUM_CLASSES, pretrained=True, drop_rate=0.2, device=device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Run logging
    run_dir = RUNS_DIR / f"resnet50_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINTS_DIR / f"resnet50_seed{seed}_best.pt"

    best_val_img_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history = []

    logging.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_patch_f1 = evaluate_patches(model, val_loader, criterion, device)
        val_img_results = evaluate_image_level(model, val_image_ds, device)
        val_img_f1 = val_img_results["macro_f1"]
        val_img_acc = val_img_results["accuracy"]

        scheduler.step()
        epoch_time = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_patch_acc": val_acc,
            "val_patch_f1": val_patch_f1,
            "val_image_acc": val_img_acc,
            "val_image_macro_f1": val_img_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": epoch_time,
        }
        history.append(log_entry)

        logging.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Image F1: {val_img_f1:.4f} (Acc: {val_img_acc:.4f})"
        )

        # Checkpoint on best Val Image Macro-F1
        if val_img_f1 > best_val_img_f1:
            best_val_img_f1 = val_img_f1
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_image_macro_f1": val_img_f1,
                    "val_image_acc": val_img_acc,
                    "class_weights": class_weights.cpu(),
                },
                best_checkpoint_path,
                is_best=True,
            )
            # Also save canonical baseline checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_image_macro_f1": val_img_f1,
                    "val_image_acc": val_img_acc,
                    "class_weights": class_weights.cpu(),
                },
                CHECKPOINTS_DIR / "resnet50_baseline_best.pt"
            )
            logging.info(f" -> Best checkpoint saved with Val Image Macro-F1: {val_img_f1:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    # Save training log
    pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)

    # Load best checkpoint for final evaluation
    logging.info(f"\n--- Loading best model from epoch {best_epoch} for final evaluation ---")
    load_checkpoint(best_checkpoint_path, model, device=str(device))

    # Evaluate on In-Domain Test
    test_eval = evaluate_image_level(model, test_image_ds, device)
    logging.info(f"In-Domain Test Macro-F1: {test_eval['macro_f1']:.4f} (Accuracy: {test_eval['accuracy']:.4f})")

    # Evaluate on External SCIN Test
    ext_eval = None
    if ext_image_ds is not None:
        ext_eval = evaluate_image_level(model, ext_image_ds, device)
        logging.info(f"SCIN External Test Macro-F1: {ext_eval['macro_f1']:.4f} (Accuracy: {ext_eval['accuracy']:.4f})")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_img_f1": best_val_img_f1,
        "test_eval": test_eval,
        "ext_eval": ext_eval,
        "history": history,
        "checkpoint_path": str(best_checkpoint_path),
    }


def generate_evaluation_report(results_by_seed: List[Dict[str, Any]], output_path: Path = REPORTS_DIR / "step9_resnet50_baseline_report.md"):
    """Compile comprehensive markdown report with seed statistics, confusion matrices, and FST breakdown."""
    lines = [
        "# Step 9: ResNet-50 Baseline Evaluation Report",
        "",
        "This report summarizes the performance of the patch-based **ResNet-50** baseline on facial dermatological conditions (`Acne`, `Rosacea`, `Dermatitis`, `Normal`).",
        "",
        "## 1. Experimental Setup & Hardware Guardrails",
        "",
        "- **Model Architecture:** `resnet50` (pretrained ImageNet-1k weights via `timm`)",
        "- **Training Strategy:** Patch-based training (224x224 anatomical face crops & close-up tiles) with aggregate mean-probability pooling at inference",
        "- **Loss Function:** Inverse-frequency class-weighted Cross-Entropy loss ($w_c = N_{total} / (C \\times N_c)$)",
        "- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-2`), Cosine Annealing learning rate schedule",
        "- **VRAM Optimizations:** PyTorch mixed precision (`torch.amp.autocast('cuda')`), batch size 32, pinned memory DataLoaders with persistent workers",
        f"- **Seeds Evaluated:** {[r['seed'] for r in results_by_seed]}",
        "",
        "## 2. Overall Performance Summary",
        "",
        "| Evaluation Split | Metric | Seed 42 | Mean ± Std | Target / Notes |",
        "|---|---|---|---|---|",
    ]

    # Compute mean +/- std
    test_f1s = [r["test_eval"]["macro_f1"] for r in results_by_seed]
    test_accs = [r["test_eval"]["accuracy"] for r in results_by_seed]

    lines.append(f"| **In-Domain Test (303 imgs)** | **Macro-F1** | {test_f1s[0]:.4f} | {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f} | Primary in-domain metric |")
    lines.append(f"| In-Domain Test (303 imgs) | Accuracy | {test_accs[0]:.4f} | {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} | Overall image accuracy |")

    if results_by_seed[0]["ext_eval"] is not None:
        ext_f1s = [r["ext_eval"]["macro_f1"] for r in results_by_seed]
        ext_accs = [r["ext_eval"]["accuracy"] for r in results_by_seed]
        lines.append(f"| **SCIN External Test (337 imgs)** | **Macro-F1** | {ext_f1s[0]:.4f} | {np.mean(ext_f1s):.4f} ± {np.std(ext_f1s):.4f} | Unseen real-world domain test |")
        lines.append(f"| SCIN External Test (337 imgs) | Accuracy | {ext_accs[0]:.4f} | {np.mean(ext_accs):.4f} ± {np.std(ext_accs):.4f} | Overall image accuracy |")

    # In-Domain Per-Class Breakdown (Seed 42)
    s42_test = results_by_seed[0]["test_eval"]
    s42_rep = s42_test["classification_report"]
    lines.extend([
        "",
        "## 3. In-Domain Test: Detailed Per-Class Breakdown (Seed 42)",
        "",
        "| Class | Precision | Recall | F1-Score | Support (Images) |",
        "|---|---|---|---|---|",
    ])
    for cls_name in CLASS_NAMES:
        if cls_name in s42_rep:
            p = s42_rep[cls_name]["precision"]
            r = s42_rep[cls_name]["recall"]
            f = s42_rep[cls_name]["f1-score"]
            s = int(s42_rep[cls_name]["support"])
            lines.append(f"| **{cls_name}** | {p:.4f} | {r:.4f} | **{f:.4f}** | {s} |")

    macro_row = s42_rep["macro avg"]
    lines.append(f"| **Macro Average** | {macro_row['precision']:.4f} | {macro_row['recall']:.4f} | **{macro_row['f1-score']:.4f}** | {int(macro_row['support'])} |")

    # In-Domain Confusion Matrix
    cm_test = s42_test["confusion_matrix"]
    lines.extend([
        "",
        "### In-Domain Test Confusion Matrix",
        "",
        "Rows: True Label | Columns: Predicted Label",
        "",
        "| True \\ Pred | " + " | ".join(CLASS_NAMES) + " |",
        "|---|---|---|---|---|",
    ])
    for i, cls_name in enumerate(CLASS_NAMES):
        row_str = " | ".join(str(cm_test[i][j]) for j in range(NUM_CLASSES))
        lines.append(f"| **{cls_name}** | {row_str} |")

    # SCIN External Test Breakdown
    if results_by_seed[0]["ext_eval"] is not None:
        s42_ext = results_by_seed[0]["ext_eval"]
        s42_ext_rep = s42_ext["classification_report"]
        cm_ext = s42_ext["confusion_matrix"]

        lines.extend([
            "",
            "## 4. SCIN Held-Out External Test Breakdown (Seed 42)",
            "",
            "> [!NOTE]",
            "> SCIN is an independent crowd-sourced dataset with smartphone photos. SCIN contains `Acne`, `Rosacea`, and `Dermatitis` (no `Normal` images).",
            "",
            "| Class | Precision | Recall | F1-Score | Support (Images) |",
            "|---|---|---|---|---|",
        ])
        for cls_name in ["Acne", "Rosacea", "Dermatitis"]:
            if cls_name in s42_ext_rep:
                p = s42_ext_rep[cls_name]["precision"]
                r = s42_ext_rep[cls_name]["recall"]
                f = s42_ext_rep[cls_name]["f1-score"]
                s = int(s42_ext_rep[cls_name]["support"])
                lines.append(f"| **{cls_name}** | {p:.4f} | {r:.4f} | **{f:.4f}** | {s} |")

        lines.extend([
            "",
            "### SCIN External Test Confusion Matrix",
            "",
            "| True \\ Pred | " + " | ".join(CLASS_NAMES) + " |",
            "|---|---|---|---|---|",
        ])
        for i, cls_name in enumerate(CLASS_NAMES):
            row_str = " | ".join(str(cm_ext[i][j]) for j in range(NUM_CLASSES))
            lines.append(f"| **{cls_name}** | {row_str} |")

        # SCIN Fitzpatrick Breakdown
        ext_preds_df = s42_ext["predictions_df"]
        manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifest" / "manifest.csv")
        scin_fst_map = dict(zip(manifest_df["image_id"], manifest_df["fst"]))
        ext_preds_df["fst"] = ext_preds_df["image_id"].map(scin_fst_map)

        lines.extend([
            "",
            "## 5. SCIN Performance Breakdown by Fitzpatrick Skin Type (FST)",
            "",
            "| Fitzpatrick Skin Type | Images | Accuracy | Correct | Incorrect |",
            "|---|---|---|---|---|",
        ])
        for fst_val, grp in ext_preds_df.groupby("fst", dropna=False):
            fst_str = str(fst_val) if pd.notna(fst_val) else "Unspecified"
            n_grp = len(grp)
            n_corr = grp["correct"].sum()
            acc_grp = n_corr / n_grp if n_grp > 0 else 0.0
            lines.append(f"| **{fst_str}** | {n_grp} | {acc_grp * 100:.1f}% | {n_corr} | {n_grp - n_corr} |")

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n" + report_text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet-50 baseline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds to run")
    args = parser.parse_args()

    results = []
    for s in args.seeds:
        res = train_baseline(
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
        )
        results.append(res)

    generate_evaluation_report(results)


if __name__ == "__main__":
    main()
