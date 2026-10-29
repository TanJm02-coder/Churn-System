"""Fairness audit: group error rates with sample sizes and bootstrap CIs.

Report & discuss only — no mitigation is applied. ``gender`` is excluded from
the model's features but still audited here. Excluding it does NOT remove proxy
effects: the model can encode group information through correlated features, and
``SeniorCitizen``/``Partner``/``Dependents`` remain features and may act as
proxies. This is discussed in the written report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import FAIRNESS_GROUPS
from churn.metrics import rate_ci


def _group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """FPR, FNR, recall, selection rate for one group slice."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tp = int(((y_pred == 1) & pos).sum())
    fn = int(((y_pred == 0) & pos).sum())
    fp = int(((y_pred == 1) & neg).sum())
    tn = int(((y_pred == 0) & neg).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    selection = (tp + fp) / len(y_true) if len(y_true) else float("nan")
    return {"recall": recall, "fpr": fpr, "fnr": fnr, "selection_rate": selection}


def audit(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: tuple[str, ...] = FAIRNESS_GROUPS,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Per-group error rates with n and bootstrap 95% CIs on each rate.

    ``df_test`` must be aligned row-wise with y_true/y_pred and contain the
    grouping columns (including the excluded ``gender``).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rows = []

    for col in groups:
        if col not in df_test.columns:
            continue
        for value in sorted(df_test[col].unique().tolist(), key=str):
            mask = (df_test[col] == value).to_numpy()
            n = int(mask.sum())
            yt, yp = y_true[mask], y_pred[mask]
            rates = _group_rates(yt, yp)

            # Bootstrap CIs for recall and selection rate (the two most
            # decision-relevant). Recall CI uses only the positives in-group.
            pos_mask = yt == 1
            recall_vals = (yp[pos_mask] == 1).astype(float)
            recall_ci = (
                rate_ci(recall_vals, n_boot=n_boot)
                if pos_mask.any()
                else (
                    float("nan"),
                    float("nan"),
                )
            )
            sel_vals = (yp == 1).astype(float)
            sel_ci = rate_ci(sel_vals, n_boot=n_boot) if n else (float("nan"), float("nan"))

            rows.append(
                {
                    "attribute": col,
                    "group": str(value),
                    "n": n,
                    **{k: round(v, 4) for k, v in rates.items()},
                    "recall_ci_lo": round(recall_ci[0], 4),
                    "recall_ci_hi": round(recall_ci[1], 4),
                    "selection_ci_lo": round(sel_ci[0], 4),
                    "selection_ci_hi": round(sel_ci[1], 4),
                }
            )

    return pd.DataFrame(rows)
