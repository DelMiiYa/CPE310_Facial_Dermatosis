"""Task 7A: Train EfficientNet-B0 Patch Classifier.

Trains an EfficientNet-B0 backbone using identical hyperparameters and data splits
as the ResNet-50 baseline to enable an exact head-to-head comparison.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train_baseline import train_baseline, generate_evaluation_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 baseline classifier")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay (default: 1e-2)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds (default: [42])")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "step10_efficientnet_b0_report.md",
        help="Path for evaluation report markdown",
    )
    args = parser.parse_args()

    results = []
    for s in args.seeds:
        res = train_baseline(
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            patience=args.patience,
            backbone="efficientnet_b0",
        )
        results.append(res)

    logging.info("EfficientNet-B0 training completed.")


if __name__ == "__main__":
    main()
