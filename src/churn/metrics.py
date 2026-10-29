"""Metric helpers: recall-at-precision, PR-AUC, and bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from churn.config import RANDOM_STATE


def recall_at_precision(y_true, y_score, target_precision: float) -> float:
    """Max recall achievable at a precision >= target_precision.

    Returns 0.0 if no threshold reaches the target precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns arrays of len n_thresholds+1.
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def core_scores(y_true, y_score) -> dict[str, float]:
    """ROC-AUC, PR-AUC, and recall at the configured precision points."""
    from churn.config import PRECISION_POINTS

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    for p in PRECISION_POINTS:
        out[f"recall@p{p}"] = recall_at_precision(y_true, y_score, p)
    return out


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Bootstrap percentile CI for a statistic over 1-D ``values``.

    ``stat_fn`` maps a resampled array to a scalar. Used for per-segment /
    per-group rate CIs. Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        stats[i] = stat_fn(sample)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def rate_ci(mask_values: np.ndarray, **kwargs) -> tuple[float, float]:
    """Bootstrap CI for a binary rate (mean of a 0/1 array)."""
    return bootstrap_ci(mask_values.astype(float), np.mean, **kwargs)
