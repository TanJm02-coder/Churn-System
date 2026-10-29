"""Customer segmentation (k-means) with profiles and per-segment performance.

k is chosen via silhouette score (with an elbow/inertia curve also recorded).
Per-segment model metrics report sample size and bootstrap CIs; small segments
are flagged as low-confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, silhouette_score

from churn.config import RANDOM_STATE
from churn.metrics import bootstrap_ci
from churn.pipeline import build_preprocessor


def choose_k(x_features: np.ndarray, k_range=range(2, 8)) -> tuple[int, dict]:
    """Pick k by max silhouette; also record inertia for an elbow view."""
    silhouettes, inertias = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(x_features)
        silhouettes[k] = float(silhouette_score(x_features, labels))
        inertias[k] = float(km.inertia_)
    best_k = max(silhouettes, key=silhouettes.get)
    return best_k, {"silhouette": silhouettes, "inertia": inertias}


def fit_segments(x_train: pd.DataFrame, k: int | None = None) -> tuple[KMeans, np.ndarray, object]:
    """Fit k-means on preprocessed training features. Returns (km, labels, preproc)."""
    preproc = build_preprocessor(x_train, scale=True)
    feats = preproc.fit_transform(x_train)
    if k is None:
        k, _ = choose_k(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(feats)
    return km, labels, preproc


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Human-readable profile per segment."""
    d = df.copy()
    d["segment"] = labels
    rows = []
    for seg, g in d.groupby("segment"):
        rows.append(
            {
                "segment": int(seg),
                "n": int(len(g)),
                "churn_rate": round(float(g["Churn"].mean()), 4),
                "avg_monthly": round(float(g["MonthlyCharges"].mean()), 2),
                "avg_tenure": round(float(g["tenure"].mean()), 1),
                "top_contract": g["Contract"].astype(str).mode().iloc[0],
                "top_payment": g["PaymentMethod"].astype(str).mode().iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def performance_by_segment(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """PR-AUC per segment with sample size and bootstrap CI.

    Small segments (few positives) yield unstable PR-AUC; the CI width and n
    make that visible. Segments with <2 classes present get NaN PR-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    labels = np.asarray(labels)
    rows = []

    for seg in sorted(np.unique(labels)):
        mask = labels == seg
        yt, ys = y_true[mask], y_score[mask]
        n = int(mask.sum())
        n_pos = int(yt.sum())
        if len(np.unique(yt)) < 2:
            pr_auc, ci = float("nan"), (float("nan"), float("nan"))
        else:
            pr_auc = float(average_precision_score(yt, ys))
            idx = np.arange(n)

            def _stat(resampled_idx, _yt=yt, _ys=ys):
                if len(np.unique(_yt[resampled_idx])) < 2:
                    return np.nan
                return average_precision_score(_yt[resampled_idx], _ys[resampled_idx])

            ci = bootstrap_ci(idx, _stat, n_boot=n_boot)
        rows.append(
            {
                "segment": int(seg),
                "n": n,
                "n_positives": n_pos,
                "pr_auc": None if np.isnan(pr_auc) else round(pr_auc, 4),
                "pr_auc_ci_lo": None if np.isnan(ci[0]) else round(ci[0], 4),
                "pr_auc_ci_hi": None if np.isnan(ci[1]) else round(ci[1], 4),
                "low_confidence": n_pos < 30,
            }
        )
    return pd.DataFrame(rows)
