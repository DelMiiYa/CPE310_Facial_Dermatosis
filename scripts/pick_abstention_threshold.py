"""Derive validation-split selective prediction abstention threshold.

Computes coverage, retention, threshold, accuracy, and Macro-F1 on the 294 validation
images from runs/ensemble_seed42/val_predictions.csv using calibrated ensemble
probabilities (T_resnet=0.60, T_effnet=1.20, w_resnet=0.40).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.calibration import predict_calibrated_ensemble, evaluate_selective_prediction

VAL_PREDICTIONS_CSV = PROJECT_ROOT / "runs" / "ensemble_seed42" / "val_predictions.csv"
CALIBRATION_W_RESNET = 0.40
CALIBRATION_T_RESNET = 0.60
CALIBRATION_T_EFFNET = 1.20


def main() -> None:
    if not VAL_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Validation predictions not found at {VAL_PREDICTIONS_CSV}")

    df = pd.read_csv(VAL_PREDICTIONS_CSV)
    print(f"Loaded {len(df)} validation predictions from {VAL_PREDICTIONS_CSV}")

    val_res = df[[f"resnet_prob_{c}" for c in CLASS_NAMES]].values.astype(np.float64)
    val_eff = df[[f"effnet_prob_{c}" for c in CLASS_NAMES]].values.astype(np.float64)
    val_targets = np.array([CLASS_TO_IDX[label] for label in df["true_label"]], dtype=np.int64)

    # Compute calibrated validation probabilities
    val_cal_probs = predict_calibrated_ensemble(
        val_res,
        val_eff,
        w_resnet=CALIBRATION_W_RESNET,
        t_resnet=CALIBRATION_T_RESNET,
        t_effnet=CALIBRATION_T_EFFNET,
    )

    coverages = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    results = evaluate_selective_prediction(
        val_cal_probs,
        val_targets,
        coverages=coverages,
        uncertainty_metric="confidence",
    )

    # Print markdown table
    print("\n### Validation Selective Prediction Performance (Calibrated Ensemble)")
    print("| Coverage (%) | Retained (k/total) | Abstention Count | Threshold (tau) | Val Accuracy (%) | Val Macro-F1 |")
    print("|---|---|---|---|---|---|")

    tau_85 = None
    for r in results:
        cov_pct = r["coverage"] * 100.0
        acc_pct = r["accuracy"] * 100.0
        f1 = r["macro_f1"]
        tau = r["threshold"]
        k = r["k"]
        total = r["total"]
        abstention_count = r["abstention_count"]
        print(f"| {cov_pct:6.1f}% | {k:3d}/{total:3d} | {abstention_count:3d} | {tau:.4f} | {acc_pct:6.2f}% | {f1:.4f} |")
        if abs(r["coverage"] - 0.85) < 1e-5:
            tau_85 = round(tau, 4)

    print("\nExact recommended constant line for 85% validation coverage:")
    print(f"ABSTENTION_THRESHOLD: float = {tau_85:.4f}")


if __name__ == "__main__":
    main()
