"""Cleaning invariants: TotalCharges blanks, dtypes, target domain, no NaNs."""

from __future__ import annotations

from churn.config import TARGET
from churn.data import clean


def test_total_charges_blank_imputed_to_zero(raw_like_df):
    cleaned = clean(raw_like_df)
    # The blank string was at tenure==0 (row 0) -> imputed to 0.0.
    assert cleaned.loc[0, "tenure"] == 0
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned["TotalCharges"].dtype == float


def test_target_is_binary_int(raw_like_df):
    cleaned = clean(raw_like_df)
    assert cleaned[TARGET].dtype.kind in ("i", "u")
    assert set(cleaned[TARGET].unique()).issubset({0, 1})


def test_no_nans_after_cleaning(raw_like_df):
    cleaned = clean(raw_like_df)
    assert cleaned.isna().sum().sum() == 0


def test_no_duplicates_dropped_silently(raw_like_df):
    # Cleaning must NOT drop rows (duplicates are reported, not removed).
    cleaned = clean(raw_like_df)
    assert len(cleaned) == len(raw_like_df)


def test_ranges_valid(raw_like_df):
    cleaned = clean(raw_like_df)
    assert (cleaned["tenure"] >= 0).all()
    assert (cleaned["MonthlyCharges"] > 0).all()
    assert (cleaned["TotalCharges"] >= 0).all()
