"""Model definitions, cross-validated comparison, and calibration.

Model selection uses stratified CV on the **training set only**. The best model
(by CV PR-AUC) is calibrated and then evaluated once on the held-out test set.
Out-of-fold (OOF) predictions on the training set are produced for the profit
threshold optimizer so the threshold is never chosen on test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from churn.config import CV_FOLDS, RANDOM_STATE
from churn.metrics import core_scores
from churn.pipeline import build_model_pipeline


def build_estimators(y_train: pd.Series) -> dict[str, tuple[object, bool]]:
    """Return {name: (estimator, needs_scaling)}.

    Class imbalance (~26.5% positive) is handled per-model: balanced class
    weights for LogReg/RF, scale_pos_weight for XGBoost.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }


def make_cv() -> StratifiedKFold:
    """Shared, seeded CV splitter used everywhere (selection + calibration)."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def calibration_method(n: int) -> str:
    """Isotonic when data is ample, else Platt/sigmoid. Shared by CV and final fit."""
    return "isotonic" if n >= 1000 else "sigmoid"


def build_calibrated_pipeline(name: str, x_train: pd.DataFrame, y_train: pd.Series):
    """A CalibratedClassifierCV wrapping the model pipeline, seeded CV.

    This is the estimator used both for the final test model AND for producing
    out-of-fold probabilities, so the threshold (chosen on OOF) and the test
    predictions live on the SAME calibrated probability scale (fixes C1).
    """
    est, scale = build_estimators(y_train)[name]
    pipe = build_model_pipeline(est, x_train, scale=scale)
    method = calibration_method(len(y_train))
    return CalibratedClassifierCV(pipe, method=method, cv=make_cv())


def cv_compare(
    x_train: pd.DataFrame, y_train: pd.Series, calibrated: bool = True
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[dict]]]:
    """Cross-validate every model on the training set.

    When ``calibrated`` is True (default), out-of-fold probabilities come from
    the *calibrated* estimator (calibration is fit within each outer fold, so no
    leakage), matching the scale of the final calibrated test model. Metrics are
    reported both as pooled-OOF and as per-fold mean +/- std.

    Returns (scores_table, oof_predictions, per_fold_scores).
    """
    cv = make_cv()
    rows = []
    oof: dict[str, np.ndarray] = {}
    per_fold: dict[str, list[dict]] = {}

    for name in build_estimators(y_train):
        if calibrated:
            estimator = build_calibrated_pipeline(name, x_train, y_train)
        else:
            est, scale = build_estimators(y_train)[name]
            estimator = build_model_pipeline(est, x_train, scale=scale)

        # cross_val_predict refits the (possibly calibrated) pipeline per fold ->
        # all preprocessing and calibration are fit on training folds only.
        proba = cross_val_predict(
            estimator, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof[name] = proba

        # Pooled-OOF scores.
        pooled = core_scores(y_train, proba)
        pooled["model"] = name
        rows.append(pooled)

        # Per-fold scores for mean +/- std (M3).
        fold_scores = []
        y_arr = y_train.to_numpy()
        for _, test_idx in cv.split(x_train, y_train):
            fold_scores.append(core_scores(y_arr[test_idx], proba[test_idx]))
        per_fold[name] = fold_scores

    table = pd.DataFrame(rows).set_index("model")
    return table, oof, per_fold


def per_fold_summary(per_fold: dict[str, list[dict]]) -> pd.DataFrame:
    """Mean +/- std across folds for each model and metric (M3)."""
    rows = []
    for name, folds in per_fold.items():
        fold_df = pd.DataFrame(folds)
        row = {"model": name}
        for col in fold_df.columns:
            row[f"{col}_mean"] = float(fold_df[col].mean())
            row[f"{col}_std"] = float(fold_df[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_best(scores: pd.DataFrame) -> str:
    """Pick the best model by CV PR-AUC (headline for imbalanced churn)."""
    return str(scores["pr_auc"].idxmax())


# Tree-based models eligible for SHAP TreeExplainer.
TREE_MODELS = ("xgboost", "random_forest")


def select_best_tree(scores: pd.DataFrame) -> str:
    """Pick the best *tree* model by CV PR-AUC (for SHAP interpretability).

    SHAP's TreeExplainer only supports tree ensembles, so even when a linear
    model wins overall, SHAP is computed on the best tree model.
    """
    tree_scores = scores.loc[scores.index.isin(TREE_MODELS)]
    return str(tree_scores["pr_auc"].idxmax())


def fit_calibrated(name: str, x_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """Fit the chosen model with probability calibration on training data only.

    Uses the shared seeded CV (M2) so calibration is reproducible and consistent
    with the OOF procedure in ``cv_compare``. The internal CV means calibration
    never sees the held-out test set.
    """
    calibrated = build_calibrated_pipeline(name, x_train, y_train)
    calibrated.fit(x_train, y_train)
    return calibrated


def fit_plain(name: str, x_train: pd.DataFrame, y_train: pd.Series):
    """Fit the uncalibrated pipeline (for before/after calibration comparison)."""
    est, scale = build_estimators(y_train)[name]
    pipe = build_model_pipeline(est, x_train, scale=scale)
    pipe.fit(x_train, y_train)
    return pipe
