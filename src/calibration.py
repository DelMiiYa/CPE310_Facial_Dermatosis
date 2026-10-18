"""Post-Hoc Confidence Calibration & Domain Adaptation Utilities.

Provides rigorous post-hoc calibration, temperature scaling, Bayes label-shift
adaptation (Saerens et al. 2002), and selective prediction evaluation under
strict zero-leakage constraints.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def compute_ece_and_brier(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE), MCE, Brier score, NLL, and entropy.

    Args:
        probs: Array of predicted probabilities of shape (N, C), each row summing to 1.
        targets: Array of integer ground-truth class labels of shape (N,) in [0, C-1].
        n_bins: Number of equal-width confidence bins in [0, 1]. Default 10.
        eps: Small epsilon for numerical stability in log operations.

    Returns:
        Dict containing:
            - 'ece': Expected Calibration Error in [0, 1]
            - 'mce': Maximum Calibration Error in [0, 1]
            - 'brier': Multi-class Brier score
            - 'nll': Negative Log-Likelihood
            - 'entropy': Mean predictive Shannon entropy in nats
            - 'accuracy': Top-1 classification accuracy
            - 'bin_edges': Array of bin edges (n_bins + 1,)
            - 'bin_confs': List of mean confidence per bin
            - 'bin_accs': List of mean accuracy per bin
            - 'bin_counts': List of sample counts per bin
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n_samples, n_classes = probs.shape

    # Ensure probabilities are properly normalized
    probs = np.clip(probs, eps, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    # Multi-class Brier score: mean sum of squared residuals
    one_hot_targets = np.zeros_like(probs)
    one_hot_targets[np.arange(n_samples), targets] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot_targets) ** 2, axis=1)))

    # Negative log-likelihood
    true_class_probs = probs[np.arange(n_samples), targets]
    nll = float(-np.mean(np.log(np.maximum(true_class_probs, eps))))

    # Predictive entropy: H(p) = - sum_c p_c log(p_c)
    sample_entropy = -np.sum(probs * np.log(probs), axis=1)
    mean_entropy = float(np.mean(sample_entropy))

    # Reliability binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: List[float] = []
    bin_accs: List[float] = []
    bin_counts: List[int] = []

    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        low = bin_edges[b]
        high = bin_edges[b + 1]

        if b == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            diff = abs(bin_acc - bin_conf)

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)

            ece += (count / n_samples) * diff
            if diff > mce:
                mce = diff
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((low + high) / 2.0))

    accuracy = float(np.mean(accuracies))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier_score,
        "nll": nll,
        "entropy": mean_entropy,
        "accuracy": accuracy,
        "bin_edges": bin_edges,
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
    }


def temperature_scale_probs(
    probs: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Scale softmax probabilities by a temperature parameter T in log-space.

    Applies:
        p_c_tilde = exp(log(max(p_c, eps)) / T) / sum_j exp(log(max(p_j, eps)) / T)

    Args:
        probs: Softmax probabilities of shape (..., C).
        temperature: Scalar temperature T > 0. T > 1 softens probabilities; T < 1 sharpens.
        eps: Small positive epsilon to avoid log(0).

    Returns:
        Temperature-scaled probabilities with same shape, summing to 1 along last axis.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    probs = np.asarray(probs, dtype=np.float64)
    # Convert probabilities to unnormalized pseudo-logits
    logits = np.log(np.maximum(probs, eps)) / float(temperature)
    # Numerically stable softmax
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    scaled_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return scaled_probs


def fit_temperature_nll(
    probs: np.ndarray,
    targets: np.ndarray,
    bounds: Tuple[float, float] = (0.2, 5.0),
    eps: float = 1e-12,
) -> float:
    """Fit optimal scalar temperature T* minimizing negative log-likelihood on validation data.

    Args:
        probs: Validation probabilities of shape (N, C).
        targets: Validation integer targets of shape (N,).
        bounds: Optimization search interval [min_T, max_T]. Default [0.2, 5.0].
        eps: Numerical epsilon.

    Returns:
        Optimal temperature T* as a float.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n_samples = len(targets)

    def nll_objective(temp: float) -> float:
        scaled = temperature_scale_probs(probs, temp, eps=eps)
        p_true = scaled[np.arange(n_samples), targets]
        return float(-np.mean(np.log(np.maximum(p_true, eps))))

    res = minimize_scalar(nll_objective, bounds=bounds, method="bounded")
    if not res.success:
        logger.warning("Temperature NLL optimization did not converge cleanly: %s", res.message)
    return float(res.x)


def fit_calibrated_ensemble(
    val_res_probs: np.ndarray,
    val_eff_probs: np.ndarray,
    val_targets: np.ndarray,
    w_resnet_candidates: Optional[List[float]] = None,
    t_resnet_candidates: Optional[List[float]] = None,
    t_effnet_candidates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Fit optimal temperature scaling and fusion weights strictly on validation data.

    Performs:
    1. Independent NLL-optimal temperature fitting (T_res_nll, T_eff_nll).
    2. Joint validation grid search over (w_resnet, T_resnet, T_effnet) to maximize
       Validation Macro-F1 (tie-broken by lowest ECE and closeness to w=0.5).

    Args:
        val_res_probs: ResNet validation probabilities (N, C).
        val_eff_probs: EfficientNet validation probabilities (N, C).
        val_targets: Integer ground truth validation labels (N,).
        w_resnet_candidates: Grid of ResNet weights in [0, 1].
        t_resnet_candidates: Grid of ResNet temperatures.
        t_effnet_candidates: Grid of EfficientNet temperatures.

    Returns:
        Dictionary containing optimal parameters and validation metrics.
    """
    val_targets = np.asarray(val_targets, dtype=np.int64)

    # 1. Independent NLL temperatures
    t_res_nll = fit_temperature_nll(val_res_probs, val_targets)
    t_eff_nll = fit_temperature_nll(val_eff_probs, val_targets)

    # Baseline uncalibrated ensemble (w=0.5, T=1.0)
    uncal_val_ens = 0.5 * val_res_probs + 0.5 * val_eff_probs
    uncal_metrics = compute_ece_and_brier(uncal_val_ens, val_targets)
    uncal_f1 = float(f1_score(val_targets, np.argmax(uncal_val_ens, axis=1), average="macro"))

    # NLL-calibrated ensemble (w=0.5, T=T_nll)
    s_val_r_nll = temperature_scale_probs(val_res_probs, t_res_nll)
    s_val_e_nll = temperature_scale_probs(val_eff_probs, t_eff_nll)
    nll_val_ens = 0.5 * s_val_r_nll + 0.5 * s_val_e_nll
    nll_metrics = compute_ece_and_brier(nll_val_ens, val_targets)
    nll_f1 = float(f1_score(val_targets, np.argmax(nll_val_ens, axis=1), average="macro"))

    # 2. Joint grid search
    if w_resnet_candidates is None:
        w_resnet_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    if t_resnet_candidates is None:
        t_resnet_candidates = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    if t_effnet_candidates is None:
        t_effnet_candidates = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]

    best_f1 = -1.0
    best_ece = 999.0
    best_params = (0.50, 1.0, 1.0)
    best_metrics: Dict[str, Any] = {}

    for w_res in w_resnet_candidates:
        w_eff = 1.0 - w_res
        for t_r in t_resnet_candidates:
            scaled_r = temperature_scale_probs(val_res_probs, t_r)
            for t_e in t_effnet_candidates:
                scaled_e = temperature_scale_probs(val_eff_probs, t_e)
                ens_probs = w_res * scaled_r + w_eff * scaled_e
                preds = np.argmax(ens_probs, axis=1)

                f1 = float(f1_score(val_targets, preds, average="macro"))
                metrics = compute_ece_and_brier(ens_probs, val_targets)
                ece = metrics["ece"]

                # Selection rule: Maximize Macro-F1, tie-break on lower ECE, tie-break on closeness to w=0.5
                is_better = False
                if f1 > best_f1 + 1e-5:
                    is_better = True
                elif abs(f1 - best_f1) <= 1e-5:
                    if ece < best_ece - 1e-4:
                        is_better = True
                    elif abs(ece - best_ece) <= 1e-4:
                        if abs(w_res - 0.50) < abs(best_params[0] - 0.50):
                            is_better = True

                if is_better:
                    best_f1 = f1
                    best_ece = ece
                    best_params = (w_res, t_r, t_e)
                    best_metrics = metrics

    opt_w_res, opt_t_res, opt_t_eff = best_params

    return {
        "t_resnet_nll": t_res_nll,
        "t_effnet_nll": t_eff_nll,
        "optimal_w_resnet": opt_w_res,
        "optimal_w_effnet": float(1.0 - opt_w_res),
        "optimal_t_resnet": opt_t_res,
        "optimal_t_effnet": opt_t_eff,
        "val_uncal_f1": uncal_f1,
        "val_uncal_ece": uncal_metrics["ece"],
        "val_uncal_acc": uncal_metrics["accuracy"],
        "val_uncal_brier": uncal_metrics["brier"],
        "val_nll_f1": nll_f1,
        "val_nll_ece": nll_metrics["ece"],
        "val_nll_acc": nll_metrics["accuracy"],
        "val_nll_brier": nll_metrics["brier"],
        "val_opt_f1": best_f1,
        "val_opt_ece": best_metrics["ece"],
        "val_opt_acc": best_metrics["accuracy"],
        "val_opt_brier": best_metrics["brier"],
    }


def predict_calibrated_ensemble(
    res_probs: np.ndarray,
    eff_probs: np.ndarray,
    w_resnet: float,
    t_resnet: float,
    t_effnet: float,
) -> np.ndarray:
    """Compute ensemble probabilities with per-backbone temperature scaling and weights.

    Args:
        res_probs: ResNet probabilities of shape (N, C).
        eff_probs: EfficientNet probabilities of shape (N, C).
        w_resnet: Weight assigned to ResNet in [0, 1].
        t_resnet: Temperature for ResNet (> 0).
        t_effnet: Temperature for EfficientNet (> 0).

    Returns:
        Calibrated ensemble probabilities of shape (N, C) summing to 1.
    """
    scaled_res = temperature_scale_probs(res_probs, t_resnet)
    scaled_eff = temperature_scale_probs(eff_probs, t_effnet)
    w_effnet = 1.0 - w_resnet
    ens_probs = w_resnet * scaled_res + w_effnet * scaled_eff
    # Ensure exact sum to 1
    return ens_probs / np.sum(ens_probs, axis=-1, keepdims=True)


def apply_bayes_prior_adjustment(
    probs: np.ndarray,
    source_priors: np.ndarray,
    target_priors: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Implement Saerens et al. (2002) Bayesian label-shift prior correction.

    Adjusts posterior probabilities p(y=c | x) when target distribution prior
    pi_target differs from source distribution prior pi_source:
        p_adj(y=c | x) = [ p(y=c | x) * (pi_target(c) / pi_source(c)) ] / Z(x)

    Supports:
    - Standard domain adaptation where all pi_target(c) > 0.
    - 3-Class Symptomatic Lesion Triage Mode where pi_target(Normal) == 0:
      Any class with pi_target(c) == 0 is zeroed out and probabilities are
      re-normalized over remaining active disease classes.

    Args:
        probs: Softmax probabilities of shape (N, C).
        source_priors: Source class prior frequencies of shape (C,), summing to 1.
        target_priors: Target class prior frequencies of shape (C,), summing to 1.
        eps: Small numerical constant.

    Returns:
        Prior-adjusted probabilities of shape (N, C), each row summing to 1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    source_priors = np.asarray(source_priors, dtype=np.float64)
    target_priors = np.asarray(target_priors, dtype=np.float64)

    # Compute ratio pi_target / pi_source
    # Where target prior is 0, the ratio is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(target_priors > 0, target_priors / np.maximum(source_priors, eps), 0.0)

    # Multiply each class probability by its prior ratio
    adjusted_unnorm = probs * ratios

    # Normalize each row
    partition = np.sum(adjusted_unnorm, axis=1, keepdims=True)
    # If a sample has zero mass across all classes (should never happen unless probs are 0), fallback to target_priors
    fallback_mask = partition <= eps
    if np.any(fallback_mask):
        logger.warning("%d samples had near-zero adjusted probability mass; reverting to target prior", np.sum(fallback_mask))
        adjusted_unnorm[fallback_mask.squeeze(1)] = target_priors
        partition = np.sum(adjusted_unnorm, axis=1, keepdims=True)

    adjusted_probs = adjusted_unnorm / partition
    return adjusted_probs


def evaluate_selective_prediction(
    probs: np.ndarray,
    targets: np.ndarray,
    coverages: Optional[List[float]] = None,
    uncertainty_metric: str = "confidence",
) -> List[Dict[str, Any]]:
    """Evaluate classification accuracy and Macro-F1 across coverage levels under rejection.

    Args:
        probs: Probabilities of shape (N, C).
        targets: Integer true labels of shape (N,).
        coverages: List of coverage fractions, e.g. [1.0, 0.9, 0.8, 0.7].
        uncertainty_metric: 'confidence' (sort by max probability descending) or
                           'entropy' (sort by predictive entropy ascending).

    Returns:
        List of dicts containing coverage, sample count, threshold, accuracy, and Macro-F1.
    """
    if coverages is None:
        coverages = [1.0, 0.90, 0.80, 0.70]

    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n_samples = len(targets)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    if uncertainty_metric == "confidence":
        score = confidences
        sorted_indices = np.argsort(-score)  # Highest confidence first
    elif uncertainty_metric == "entropy":
        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0)
        p_clipped = p_clipped / np.sum(p_clipped, axis=1, keepdims=True)
        entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)
        score = entropy
        sorted_indices = np.argsort(score)  # Lowest entropy first
    else:
        raise ValueError(f"Unknown uncertainty_metric: {uncertainty_metric}")

    results = []
    for cov in coverages:
        k = max(1, int(np.round(cov * n_samples)))
        sel_idx = sorted_indices[:k]
        sub_targets = targets[sel_idx]
        sub_preds = predictions[sel_idx]

        acc = float(accuracy_score(sub_targets, sub_preds))
        macro_f1 = float(f1_score(sub_targets, sub_preds, average="macro", zero_division=0))
        tau = float(score[sorted_indices[k - 1]])

        results.append({
            "coverage": cov,
            "k": k,
            "total": n_samples,
            "threshold": tau,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "abstention_count": n_samples - k,
        })

    return results
