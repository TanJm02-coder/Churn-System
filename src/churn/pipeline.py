"""Preprocessing pipeline builders.

The preprocessing lives inside an sklearn ``Pipeline`` so that, under
cross-validation, every transformer is fit on training folds only. This is the
structural guarantee against preprocessing leakage; a unit test additionally
verifies it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for a feature frame."""
    numeric = x.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(x: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer.

    - Numeric: median impute (defensive) + optional standardize.
    - Categorical: one-hot with ``handle_unknown="ignore"`` so unseen
      categories at predict time do not error.

    ``scale`` is True for linear models (LogReg) and False for tree models,
    which are scale-invariant.
    """
    numeric, categorical = split_feature_types(x)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric_tf = Pipeline(num_steps)

    categorical_tf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", categorical_tf, categorical),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator, x: pd.DataFrame, scale: bool) -> Pipeline:
    """Wrap preprocessing + estimator in a single Pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(x, scale=scale)),
            ("model", estimator),
        ]
    )
