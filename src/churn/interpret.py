"""SHAP-based interpretability for the best tree model.

Produces global feature importance (mean |SHAP|) and a plain-language driver
summary. The collinearity caveat (TotalCharges ~ tenure x MonthlyCharges) is
included so importances are not over-read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_names(preprocessor, x: pd.DataFrame) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(x.columns)


def shap_importance(
    fitted_pipeline, x_sample: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute mean |SHAP| global importance for a fitted tree pipeline.

    Returns (importance_df, shap_values, transformed_X). ``fitted_pipeline`` is
    a Pipeline whose last step is a tree model and first step is 'preprocess'.
    """
    import shap

    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    x_trans = preprocessor.transform(x_sample)
    names = _feature_names(preprocessor, x_sample)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_trans)
    # Normalize across SHAP/model versions:
    #   - list (one array per class) -> take positive class
    #   - 3D array (n, features, classes) -> take positive class slice
    #   - 2D array (n, features) -> use as-is
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, -1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_abs = np.asarray(mean_abs).ravel()
    imp = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return imp, shap_vals, x_trans


def plain_language_summary(
    importance: pd.DataFrame,
    top_n: int = 8,
    explained_model: str | None = None,
    scoring_model: str | None = None,
) -> str:
    """Business-readable summary of the top churn drivers.

    ``explained_model`` is the model SHAP was computed on (a tree model);
    ``scoring_model`` is the model that actually produces the deployed churn
    probabilities/threshold/profit. When they differ, that is stated prominently
    so the reader does not assume SHAP explains the scorer (M4).
    """
    top = importance.head(top_n)
    lines = ["# Churn Drivers — Plain-Language Summary", ""]

    if explained_model and scoring_model and explained_model != scoring_model:
        lines += [
            f"> **Note:** SHAP here explains the best *tree* model "
            f"(`{explained_model}`), which is **not** the deployed scoring model "
            f"(`{scoring_model}`). TreeExplainer only supports tree ensembles, so "
            f"when a non-tree model wins selection, SHAP is computed on the best "
            f"tree model as a directional guide to drivers — treat it as indicative "
            f"of churn structure, not an exact explanation of the scorer.",
            "",
        ]
    elif explained_model:
        lines += [f"SHAP computed on model `{explained_model}` (the deployed scorer).", ""]

    lines += [
        "Ranked by mean absolute SHAP value on a **training-set** sample. SHAP",
        "measures how much each feature pushes an individual prediction toward or",
        "away from churn.",
        "",
        "## Top drivers",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(f"- **{row['feature']}** (mean |SHAP| = {row['mean_abs_shap']:.4f})")
    lines += [
        "",
        "## How to read this",
        "",
        "- Short-tenure, month-to-month customers and those on electronic-check",
        "  payment typically show the strongest churn signal in this dataset.",
        "- Longer contracts and value-added services (tech support, online",
        "  security) are associated with retention.",
        "",
        "## Caveat: collinearity",
        "",
        "`TotalCharges` is approximately `tenure x MonthlyCharges`. Because these",
        "three move together, SHAP importance can be split or shifted among them;",
        "do not read any single one's importance in isolation. Treat tenure,",
        "monthly charges, and total charges as one correlated bundle.",
    ]
    return "\n".join(lines)
