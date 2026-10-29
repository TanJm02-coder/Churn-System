"""Final evaluation helpers: calibration curves and the model comparison report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from churn.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402


def plot_calibration(
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    fname: str = "calibration.png",
) -> dict[str, float]:
    """Reliability diagram for one or more probability vectors. Returns Brier scores."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    briers = {}
    for name, proba in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
        briers[name] = float(brier_score_loss(y_true, proba))
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability) curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=110)
    plt.close(fig)
    return briers


def write_model_comparison(
    cv_scores: pd.DataFrame,
    fold_summary: pd.DataFrame,
    best_name: str,
    test_scores: dict[str, float],
    briers: dict[str, float],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Comparison",
        "",
        "## Methodology",
        "",
        "Model selection, calibration, and the profit threshold are all decided",
        "using **stratified 5-fold CV on the training set only** (seeded). Out-of-",
        "fold probabilities are produced from the **calibrated** estimator, so the",
        "threshold chosen on OOF transfers to the calibrated test predictions on",
        "the same scale. The held-out 20% test set is scored **once** to produce",
        "the headline numbers. CV scores drive selection; **test-set numbers are",
        "the headline.**",
        "",
        "## Cross-validation scores — pooled out-of-fold (training set)",
        "",
        cv_scores.round(4).to_markdown(),
        "",
        "## Cross-validation scores — per-fold mean ± std (training set)",
        "",
        fold_summary.round(4).to_markdown(),
        "",
        f"**Selected model (max CV PR-AUC): `{best_name}`**",
        "",
        "## Held-out test-set scores (headline)",
        "",
    ]
    for k, v in test_scores.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines += ["", "## Calibration (Brier score, lower is better)", ""]
    for k, v in briers.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines += [
        "",
        "See `figures/calibration.png` for reliability curves (before/after",
        "calibration).",
    ]
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines))
