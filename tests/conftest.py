"""Shared fixtures. Ensures src/ is importable and provides a tiny toy frame."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """A tiny frame shaped like the raw Telco data, incl. a blank TotalCharges."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
            "tenure": [0, 5, 24, 1, 60, 12],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No", "No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "No", "Yes", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No", "Yes", "No", "No"],
            "DeviceProtection": ["No", "Yes", "No", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "No", "Yes", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No", "No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No", "No", "Yes", "No"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "Yes", "No", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Credit card (automatic)",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 70.70, 20.10, 55.00, 99.90, 45.00],
            # Note the blank string at tenure==0 (index 0).
            "TotalCharges": [" ", "353.5", "482.4", "55.0", "5994.0", "540.0"],
            "Churn": ["No", "Yes", "No", "Yes", "No", "No"],
        }
    )
