"""Guard against duplicate rows straddling the train/test split.

The cleaned Telco data contains fully-identical feature+label rows (reported,
not dropped, because customerID is absent so they may be distinct customers). If
an identical twin landed in both train and test, the test metrics would be
optimistically biased by memorization. This test asserts that does not happen
under the pinned seed; if the seed changes and this fails, deduplicate before
splitting or re-document the effect in the README Limitations.
"""

from __future__ import annotations

import numpy as np

from churn.data import clean, deduplicate_for_split, load_raw
from churn.split import load_split_indices, stratified_split


def test_no_duplicate_group_straddles_split():
    """After dedup-before-split, no identical row can straddle train/test."""
    df = clean(load_raw())
    df_model, n_dropped = deduplicate_for_split(df)
    assert n_dropped > 0, "expected some exact duplicates to be dropped"

    # Persist the canonical split indices for the deduped modeling frame.
    stratified_split(df_model, persist=True)
    split = load_split_indices()
    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    feature_cols = list(df_model.columns)
    dup_mask = df_model.duplicated(keep=False)

    # After dedup there should be no exact duplicates left at all.
    assert int(dup_mask.sum()) == 0, "duplicates remain after deduplicate_for_split"

    straddling_groups = 0
    for _, grp in df_model[dup_mask].groupby(feature_cols, observed=True):
        idxs = grp.index.tolist()
        in_train = any(i in train_set for i in idxs)
        in_test = any(i in test_set for i in idxs)
        if in_train and in_test:
            straddling_groups += 1

    assert straddling_groups == 0, (
        f"{straddling_groups} duplicate group(s) straddle the split after dedup"
    )


def test_cleaning_reports_duplicates_without_dropping():
    """Cleaning keeps duplicates (reported-not-dropped); dedup is split-only."""
    df = clean(load_raw())
    assert int(df.duplicated().sum()) > 0
    assert len(df) == len(load_raw())
    assert bool(np.any(df.duplicated(keep=False)))

    # The dedup step is a separate, explicit modeling-split concern.
    df_model, n_dropped = deduplicate_for_split(df)
    assert len(df_model) == len(df) - n_dropped
    assert int(df_model.duplicated().sum()) == 0
