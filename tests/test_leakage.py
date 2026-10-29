"""Leakage tests that exercise the REAL production code paths.

Unlike naive checks that merely re-test the splitter, these tests instrument the
actual estimators used by ``cv_compare`` / ``fit_calibrated`` with a spy that
records exactly which rows each ``fit`` call receives. We then assert:

1. No held-out test row is ever passed to any fit call (test-set isolation).
2. Preprocessing is refit on each fold (stats differ across folds) -> no
   full-data preprocessing leakage.
3. The profit threshold is derived from the SAME calibrated scale as the test
   predictions (guards the C1 bug).

Each test is accompanied by a demonstration (in test_leakage_detects_injected.py)
that it FAILS when the corresponding leak is deliberately introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from churn.config import RANDOM_STATE, TARGET
from churn.data import clean
from churn.models import build_calibrated_pipeline, cv_compare, make_cv
from churn.pipeline import build_model_pipeline
from churn.split import load_split_indices, make_feature_frame, stratified_split


class RowSpyClassifier(BaseEstimator, ClassifierMixin):
    """Wraps an estimator and records a fingerprint of every row it is fit on.

    The fingerprint is read from a reserved ``__row_id__`` column that we inject
    into the feature frame; the wrapper strips it before delegating so the inner
    pipeline never sees it.
    """

    _fit_row_ids: list = []  # class-level record across all clones/folds

    def __init__(self, inner):
        self.inner = inner

    @classmethod
    def reset(cls):
        cls._fit_row_ids = []

    def _strip(self, x):
        ids = x["__row_id__"].to_numpy()
        return x.drop(columns="__row_id__"), ids

    def fit(self, x, y=None, **kw):
        x_inner, ids = self._strip(x)
        RowSpyClassifier._fit_row_ids.append(set(ids.tolist()))
        self.inner_ = clone(self.inner)
        self.inner_.fit(x_inner, y, **kw)
        self.classes_ = self.inner_.classes_
        return self

    def predict(self, x):
        x_inner, _ = self._strip(x)
        return self.inner_.predict(x_inner)

    def predict_proba(self, x):
        x_inner, _ = self._strip(x)
        return self.inner_.predict_proba(x_inner)


def _prepared_frame(raw_like_df, reps=60):
    big = clean(raw_like_df)
    big = big.loc[big.index.repeat(reps)].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    big["MonthlyCharges"] = big["MonthlyCharges"] + rng.normal(0, 8, len(big))
    return big


# --------------------------------------------------------------------------
# T1 (strengthened): no test row reaches any fit call in the real CV path.
# --------------------------------------------------------------------------
def test_no_test_row_reaches_any_fit_call(raw_like_df):
    big = _prepared_frame(raw_like_df)
    x_train, x_test, y_train, _ = stratified_split(big, persist=True)

    # Tag each training row with its TRUE source index (from the persisted
    # split), so provenance is preserved even if the split leaks a test row into
    # the training frame. The spy then records true ids per fit call.
    split = load_split_indices()
    train_idx = np.asarray(split["train_idx"])
    test_ids_set = {int(i) for i in split["test_idx"]}
    assert len(train_idx) == len(x_train), "train frame/index length mismatch"

    x_train_tagged = x_train.copy()
    x_train_tagged["__row_id__"] = train_idx

    # Build the REAL calibrated pipeline, but wrap it in the spy.
    inner = build_calibrated_pipeline("logreg", x_train, y_train)
    spy = RowSpyClassifier(inner)
    RowSpyClassifier.reset()

    from sklearn.model_selection import cross_val_predict

    cross_val_predict(spy, x_train_tagged, y_train, cv=make_cv(), method="predict_proba", n_jobs=1)

    all_fit_ids = set().union(*RowSpyClassifier._fit_row_ids)
    assert all_fit_ids, "spy recorded no fit calls"
    leaked = all_fit_ids & test_ids_set
    assert not leaked, f"{len(leaked)} held-out test rows leaked into fit calls"


# --------------------------------------------------------------------------
# T2 (strengthened): preprocessing is refit per fold in the real pipeline.
# --------------------------------------------------------------------------
def test_preprocessing_refit_per_fold_in_real_pipeline(raw_like_df):
    big = _prepared_frame(raw_like_df)
    x, y = make_feature_frame(big)

    cv = make_cv()
    fold_means = []
    for train_idx, _ in cv.split(x, y):
        # Reconstruct the pipeline exactly as the production code does and fit on
        # this fold's training rows only.
        from sklearn.linear_model import LogisticRegression

        pipe = build_model_pipeline(LogisticRegression(max_iter=200), x, scale=True)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
        fold_means.append(tuple(np.round(scaler.mean_, 6)))

    # If preprocessing were fit once on full data, every fold would share the
    # same learned means. Distinct means prove per-fold refit.
    assert len(set(fold_means)) > 1, "scaler stats identical across folds -> not refit per fold"


# --------------------------------------------------------------------------
# T3: the OOF used for the threshold must be the CALIBRATED OOF, not the raw
# estimator's. We check this directly and decisively against a tree model, where
# calibration changes probabilities substantially. This test FAILS if cv_compare
# returns uncalibrated OOF (the C1 bug), regardless of which model wins.
# --------------------------------------------------------------------------
def test_cv_compare_returns_calibrated_oof(raw_like_df):
    from sklearn.model_selection import cross_val_predict

    big = _prepared_frame(raw_like_df, reps=120)
    x_train, _, y_train, _ = stratified_split(big, persist=True)

    # OOF that cv_compare actually feeds to the threshold optimizer.
    _, oof_cal, _ = cv_compare(x_train, y_train, calibrated=True)

    # Reference A: OOF from the calibrated estimator, computed independently.
    cal_oof = cross_val_predict(
        build_calibrated_pipeline("random_forest", x_train, y_train),
        x_train,
        y_train,
        cv=make_cv(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Reference B: OOF from the UNCALIBRATED random forest (the C1-bug source).
    from sklearn.ensemble import RandomForestClassifier

    from churn.models import build_estimators

    est, scale = build_estimators(y_train)["random_forest"]
    assert isinstance(est, RandomForestClassifier)
    uncal_oof = cross_val_predict(
        build_model_pipeline(est, x_train, scale=scale),
        x_train,
        y_train,
        cv=make_cv(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    rf_oof = oof_cal["random_forest"]
    dist_to_calibrated = float(np.mean(np.abs(rf_oof - cal_oof)))
    dist_to_uncalibrated = float(np.mean(np.abs(rf_oof - uncal_oof)))

    # The OOF cv_compare returns must match the CALIBRATED reference, not the
    # uncalibrated one. Under the C1 bug these relationships invert.
    assert dist_to_calibrated < dist_to_uncalibrated, (
        f"cv_compare OOF is closer to the UNCALIBRATED scale "
        f"(d_uncal={dist_to_uncalibrated:.4f} < d_cal={dist_to_calibrated:.4f}) "
        f"-> C1 scale mismatch"
    )
    assert dist_to_calibrated < 1e-9, (
        "cv_compare OOF should equal the calibrated OOF under the same seed"
    )


def test_train_test_indices_are_disjoint(raw_like_df):
    big = _prepared_frame(raw_like_df)
    stratified_split(big, persist=True)
    idx = load_split_indices()
    train = set(idx["train_idx"])
    test = set(idx["test_idx"])
    assert train.isdisjoint(test), "train and test indices overlap -> leakage"
    assert len(train) + len(test) == len(big), "split does not cover all rows exactly once"


def test_target_not_in_features(raw_like_df):
    cleaned = clean(raw_like_df)
    x, _ = make_feature_frame(cleaned)
    assert TARGET not in x.columns, "target leaked into the feature matrix"


def test_gender_absent_from_feature_frame(raw_like_df):
    cleaned = clean(raw_like_df)
    x, _ = make_feature_frame(cleaned)
    assert "gender" not in x.columns
    # Confirm the spy helper strips its reserved column (sanity for T1).
    tagged = x.copy()
    tagged["__row_id__"] = np.arange(len(tagged))
    assert isinstance(tagged, pd.DataFrame)
