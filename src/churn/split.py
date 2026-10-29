"""Stratified train/test split with persisted, reproducible indices.

The test set is created once and **never** used for model selection,
calibration, or threshold choice (those use CV on the training set only). The
split indices are persisted so every downstream step and the leakage test use
the exact same partition.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    EXCLUDED_FEATURES,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

SPLIT_INDICES_PATH = MODELS_DIR / "split_indices.json"


def make_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with excluded sensitive features (e.g. gender) removed.

    ``gender`` is dropped from the model feature matrix by design; it is kept in
    the full frame elsewhere for the fairness audit. See REQUIREMENTS.md 4.1.
    """
    y = df[TARGET].astype(int)
    drop_cols = [TARGET, *[c for c in EXCLUDED_FEATURES if c in df.columns]]
    x = df.drop(columns=drop_cols)
    return x, y


def stratified_split(
    df: pd.DataFrame, persist: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on the target, seeded. Persists indices."""
    y = df[TARGET].astype(int)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_idx.sort()
    test_idx.sort()

    if persist:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SPLIT_INDICES_PATH.write_text(
            json.dumps(
                {
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                },
            )
        )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    x_train, y_train = make_feature_frame(df_train)
    x_test, y_test = make_feature_frame(df_test)
    return x_train, x_test, y_train, y_test


def load_split_indices() -> dict:
    """Load persisted split indices (used by the test-set isolation test)."""
    if not SPLIT_INDICES_PATH.exists():
        raise FileNotFoundError("Split indices not found; run the pipeline first.")
    return json.loads(SPLIT_INDICES_PATH.read_text())
