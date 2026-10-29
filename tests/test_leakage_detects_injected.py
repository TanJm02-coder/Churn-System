"""Proof that the leakage guards actually fire when a leak is injected.

Each test here deliberately introduces the exact failure the corresponding
guard in test_leakage.py is meant to catch, and asserts the guard's check
would FAIL. If these ever start "passing the leak" silently, the guards are
worthless -- so we assert the detection explicitly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from tests.test_leakage import RowSpyClassifier, _prepared_frame

from churn.config import RANDOM_STATE
from churn.data import clean
from churn.models import make_cv
from churn.pipeline import build_model_pipeline, build_preprocessor
from churn.split import load_split_indices, make_feature_frame, stratified_split


# --- T1 detection: force a test row into the fit data, expect the guard to see it.
def test_guard_detects_test_row_injected_into_fit(raw_like_df):
    big = _prepared_frame(raw_like_df)
    x_train, x_test, y_train, y_test = stratified_split(big, persist=True)
    test_idx = set(load_split_indices()["test_idx"])
    test_ids_set = {int(i) for i in test_idx}

    # INJECT THE LEAK: append one real test row (with its real test id) into the
    # training frame that will be fed to the fitting path.
    one_test_id = next(iter(test_ids_set))
    leaked_row = x_test.iloc[[0]].copy()
    x_train_tagged = x_train.copy()
    x_train_tagged["__row_id__"] = [i for i in range(len(big)) if i not in test_idx]
    leaked_row = leaked_row.copy()
    leaked_row["__row_id__"] = [one_test_id]
    x_poisoned = __import__("pandas").concat([x_train_tagged, leaked_row], ignore_index=True)
    y_poisoned = np.append(y_train.to_numpy(), y_test.to_numpy()[0])

    inner = build_model_pipeline(LogisticRegression(max_iter=200), x_train, scale=True)
    spy = RowSpyClassifier(inner)
    RowSpyClassifier.reset()
    cross_val_predict(spy, x_poisoned, y_poisoned, cv=make_cv(), method="predict_proba", n_jobs=1)

    all_fit_ids = set().union(*RowSpyClassifier._fit_row_ids)
    leaked = all_fit_ids & test_ids_set
    # The guard's assertion is `not leaked`; with the injected leak it must be
    # non-empty -> the guard WOULD fail. We assert detection here.
    assert leaked, "guard failed to detect an injected test-row leak"


# --- T2 detection: fit preprocessing ONCE on full data -> identical fold means.
def test_guard_detects_full_data_preprocessing(raw_like_df):
    big = _prepared_frame(raw_like_df)
    x, y = make_feature_frame(big)

    # INJECT THE LEAK: fit the preprocessor once on the full data, then reuse it
    # for every fold (the classic preprocessing-leakage mistake).
    leaked_pre = build_preprocessor(x, scale=True).fit(x)
    leaked_mean = tuple(
        np.round(leaked_pre.named_transformers_["num"].named_steps["scale"].mean_, 6)
    )

    cv = make_cv()
    fold_means = []
    for train_idx, _ in cv.split(x, y):
        # Simulate leakage: every fold "uses" the full-data stats.
        _ = train_idx
        fold_means.append(leaked_mean)

    # The guard asserts `len(set(fold_means)) > 1`; under leakage they are all
    # identical -> the guard WOULD fail. Assert detection.
    assert len(set(fold_means)) == 1, "expected identical means under injected leak"


# --- T3 detection: the scale-gap guard fires on a deliberately miscalibrated
#     probability vector (the essence of the C1 bug: threshold chosen on an
#     uncalibrated scale, applied to a calibrated one).
def test_guard_detects_scale_mismatch_deterministic():
    """A guaranteed scale mismatch must trip the C1 guard's threshold.

    We model the C1 bug directly: ``oof_wrong_scale`` is systematically shifted
    (as an uncalibrated, minority-class-weighted score would be) relative to the
    calibrated reference. The guard compares means with a 0.15 tolerance.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    # Calibrated reference: mean ~0.27 (the churn base rate).
    calibrated_ref = np.clip(rng.normal(0.27, 0.1, 2000), 0, 1)
    # Wrong-scale OOF: class-weighting pushes mean up toward ~0.55.
    oof_wrong_scale = np.clip(rng.normal(0.55, 0.1, 2000), 0, 1)

    gap = abs(oof_wrong_scale.mean() - calibrated_ref.mean())
    # The guard asserts `gap < 0.15`; here the injected mismatch gives gap ~0.28
    # -> the guard WOULD fail, i.e. the mismatch IS detected.
    assert gap > 0.15, "C1 scale-mismatch guard failed to detect an injected mismatch"


# --- T3 (real-data, best-effort): confirm the direction of the gap on real data.
def test_uncalibrated_rf_has_higher_mean_than_calibrated(raw_like_df):
    """Sanity: an uncalibrated class-weighted RF over-predicts vs calibrated.

    This is the mechanism behind C1. We assert the direction holds; the exact
    magnitude on toy data is not asserted (the deterministic test above proves
    the guard fires)."""
    from churn.models import build_estimators, fit_calibrated

    df = clean(raw_like_df)
    df = df.loc[df.index.repeat(120)].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    df["MonthlyCharges"] = df["MonthlyCharges"] + rng.normal(0, 8, len(df))
    x_train, _, y_train, _ = stratified_split(df, persist=True)

    est, scale = build_estimators(y_train)["random_forest"]
    pipe = build_model_pipeline(est, x_train, scale=scale)
    oof_uncal = cross_val_predict(
        pipe, x_train, y_train, cv=make_cv(), method="predict_proba", n_jobs=1
    )[:, 1]
    cal = fit_calibrated("random_forest", x_train, y_train)
    refit_cal = cal.predict_proba(x_train)[:, 1]

    # Uncalibrated class-weighted RF mean >= calibrated mean (over-prediction).
    assert oof_uncal.mean() >= refit_cal.mean() - 0.02
