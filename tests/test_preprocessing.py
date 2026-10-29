"""Preprocessing checks: gender excluded, unseen categories handled, no NaN leak."""

from __future__ import annotations

import numpy as np

from churn.config import EXCLUDED_FEATURES
from churn.data import clean
from churn.pipeline import build_preprocessor
from churn.split import make_feature_frame


def test_gender_excluded_from_features(raw_like_df):
    cleaned = clean(raw_like_df)
    x, _ = make_feature_frame(cleaned)
    for col in EXCLUDED_FEATURES:
        assert col not in x.columns, f"{col} must be excluded from model features"


def test_onehot_handles_unseen_categories(raw_like_df):
    cleaned = clean(raw_like_df)
    x, _ = make_feature_frame(cleaned)
    pre = build_preprocessor(x, scale=False)
    # Fit on rows 1..n, then transform row 0 which can carry a new category.
    train = x.iloc[1:].copy()
    test = x.iloc[:1].copy()
    test = test.copy()
    test["Contract"] = test["Contract"].astype(str)
    test.loc[test.index[0], "Contract"] = "Totally New Contract"
    pre.fit(train)
    out = pre.transform(test)  # must not raise thanks to handle_unknown="ignore"
    assert out.shape[0] == 1
    assert not np.isnan(out).any()


def test_no_nan_leaks_through_transform(raw_like_df):
    cleaned = clean(raw_like_df)
    x, _ = make_feature_frame(cleaned)
    pre = build_preprocessor(x, scale=True)
    out = pre.fit_transform(x)
    assert not np.isnan(out).any()
