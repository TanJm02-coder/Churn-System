"""Central configuration: seeds, paths, and economic assumptions.

Every stochastic component in the project is seeded from ``RANDOM_STATE`` so
results are reproducible. Economic assumptions live in ``ProfitAssumptions`` and
are the single source of truth; the Streamlit app overrides them interactively.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_RAW / "telco_churn.csv"
CLEAN_PARQUET = DATA_PROCESSED / "telco_clean.parquet"
MANIFEST = DATA_RAW / "DATA_MANIFEST.json"

# --- Data facts ------------------------------------------------------------
OPENML_DATA_ID = 42178
TARGET = "Churn"
# NOTE: OpenML 42178 marks ``customerID`` as an ignore_attribute and does not
# include it in the exported frame, so there is no identifier column to drop.
ID_COL = "customerID"

# Sensitive attribute EXCLUDED from model features by design, but retained for
# the fairness audit. See REQUIREMENTS.md section 4.1 and the README.
EXCLUDED_FEATURES = ("gender",)

# Groups audited for fairness (report & discuss only, no mitigation).
FAIRNESS_GROUPS = ("gender", "SeniorCitizen", "Partner", "Dependents")

# Precision points at which recall is reported.
PRECISION_POINTS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class ProfitAssumptions:
    """Documented economic assumptions for the profit layer.

    NONE of these are measured from the data; they are stated assumptions. The
    profit headline is therefore a *simulated expected value*, not measured
    profit. All values are adjustable in the Streamlit app.
    """

    # Offer cost per targeted customer (USD).
    offer_cost: float = 50.0
    # Probability a targeted churner is actually retained.
    offer_success_rate: float = 0.30
    # Gross margin on monthly revenue (telco midpoint; documented assumption).
    gross_margin: float = 0.30
    # CLV horizon cap in months (keeps CLV finite and conservative).
    expected_lifetime_months: int = 24
    # Annual discount rate (cost of capital); converted to monthly internally.
    annual_discount_rate: float = 0.10

    @property
    def monthly_discount_rate(self) -> float:
        return self.annual_discount_rate / 12.0

    def with_overrides(self, **kwargs: float) -> ProfitAssumptions:
        """Return a copy with the given fields overridden (used by the app)."""
        return replace(self, **kwargs)


DEFAULT_ASSUMPTIONS = ProfitAssumptions()
