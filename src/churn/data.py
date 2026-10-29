"""Data loading and cleaning with documented quality checks.

The cleaning path is deterministic and enforced by assertions so the same raw
file always produces the same clean frame. ``clean()`` returns the cleaned
DataFrame; ``run_cleaning_report()`` additionally writes a Markdown quality
report. All invariants are covered by unit tests in ``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn.config import (
    CLEAN_PARQUET,
    ID_COL,
    RAW_CSV,
    REPORTS_DIR,
    TARGET,
)

# Columns that are semantically binary Yes/No.
_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


@dataclass
class CleaningReport:
    """Structured record of what cleaning did, for the quality report."""

    n_rows_raw: int = 0
    n_cols_raw: int = 0
    n_rows_clean: int = 0
    total_charges_blanks: int = 0
    total_charges_blank_tenure_values: list = field(default_factory=list)
    duplicate_rows: int = 0
    missing_after: dict = field(default_factory=dict)
    cardinality: dict = field(default_factory=dict)
    collinearity: dict = field(default_factory=dict)
    assertions_passed: list = field(default_factory=list)


def load_raw() -> pd.DataFrame:
    """Load the raw downloaded CSV."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(RAW_CSV)


def clean(df: pd.DataFrame | None = None, report: CleaningReport | None = None) -> pd.DataFrame:
    """Clean the raw Telco frame. Pure and deterministic.

    Steps (each documented in the quality report):
      1. Drop ``customerID`` if present (identifier; OpenML usually omits it).
      2. Coerce ``TotalCharges`` to float; the 11 blank strings correspond to
         ``tenure == 0`` (brand-new customers) and are imputed to 0.0 — a
         0-tenure customer has accrued no total charges.
      3. Map target ``Churn`` Yes/No -> 1/0 (int).
      4. Normalize binary Yes/No columns to a stable categorical dtype.
      5. Sanity assertions (ranges, target domain).

    Duplicate rows are *reported but not dropped*: without ``customerID`` an
    identical feature+label row is plausibly a distinct real customer, so
    dropping risks discarding genuine data. This decision is documented.
    """
    if df is None:
        df = load_raw()
    df = df.copy()

    if report is not None:
        report.n_rows_raw = len(df)
        report.n_cols_raw = df.shape[1]

    # 1. Drop identifier if present.
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # 1b. Strip surrounding single quotes that the OpenML ARFF export wraps
    #     around multi-word categorical values (e.g. "'Fiber optic'"). Cosmetic
    #     but affects one-hot feature names and every report.
    obj_like = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    for col in obj_like:
        df[col] = df[col].astype(str).str.strip().str.strip("'")

    # 2. TotalCharges -> numeric; blanks (tenure==0) -> 0.0.
    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = coerced.isna()
    if report is not None:
        report.total_charges_blanks = int(blank_mask.sum())
        report.total_charges_blank_tenure_values = sorted(
            df.loc[blank_mask, "tenure"].unique().tolist()
        )
    coerced = coerced.fillna(0.0)
    df["TotalCharges"] = coerced.astype(float)

    # 3. Target -> int 0/1.
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)

    # 4. Normalize binary Yes/No columns to category dtype.
    for col in _YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().astype("category")

    # SeniorCitizen is 0/1 int already; keep as int.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Remaining string/object columns -> category.
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    for col in str_cols:
        df[col] = df[col].astype("category")

    # 5. Sanity assertions.
    assert (df["tenure"] >= 0).all(), "tenure has negative values"
    assert (df["MonthlyCharges"] > 0).all(), "MonthlyCharges has non-positive values"
    assert (df["TotalCharges"] >= 0).all(), "TotalCharges has negative values"
    assert set(df[TARGET].unique()).issubset({0, 1}), "target not binary"
    assert df.isna().sum().sum() == 0, "NaNs remain after cleaning"

    if report is not None:
        report.n_rows_clean = len(df)
        report.duplicate_rows = int(df.duplicated().sum())
        report.missing_after = df.isna().sum().to_dict()
        report.cardinality = {
            c: int(df[c].nunique()) for c in df.select_dtypes(include="category").columns
        }
        trio = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
        report.collinearity = {
            "tenure~TotalCharges": round(float(trio.loc["tenure", "TotalCharges"]), 3),
            "MonthlyCharges~TotalCharges": round(
                float(trio.loc["MonthlyCharges", "TotalCharges"]), 3
            ),
            "tenure~MonthlyCharges": round(float(trio.loc["tenure", "MonthlyCharges"]), 3),
        }
        report.assertions_passed = [
            "tenure >= 0",
            "MonthlyCharges > 0",
            "TotalCharges >= 0",
            "target in {0,1}",
            "no NaNs after cleaning",
        ]

    return df


def deduplicate_for_split(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate feature+label rows BEFORE the train/test split.

    Rationale: identical rows that land on both sides of the split cause
    memorized-row leakage into the test evaluation (an audit finding). Because
    OpenML omits ``customerID``, identical rows cannot be distinguished as
    separate customers for modeling purposes, so for split integrity we keep one
    representative per identical group. Returns (deduped_df, n_dropped).

    Note: the cleaning *report* still counts and documents duplicates on the full
    data; this dedup is a modeling-split concern, applied only here.
    """
    before = len(df)
    deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)


def run_cleaning_report() -> tuple[pd.DataFrame, CleaningReport]:
    """Clean and write both the parquet and the Markdown quality report."""
    report = CleaningReport()
    raw = load_raw()
    df = clean(raw, report=report)

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report_md(report)
    return df, report


def _write_report_md(r: CleaningReport) -> None:
    churn_note = ""
    lines = [
        "# Data Quality Report",
        "",
        "Generated by `src/churn/data.py`. The raw file is the OpenML Telco",
        "Customer Churn dataset (id 42178). See `data/raw/DATA_MANIFEST.json`",
        "for the exact version and checksum.",
        "",
        "## Row / column counts",
        "",
        f"- Raw: **{r.n_rows_raw}** rows x **{r.n_cols_raw}** columns",
        f"- Clean: **{r.n_rows_clean}** rows",
        "",
        "## `TotalCharges` handling",
        "",
        f"- Blank values coerced from string: **{r.total_charges_blanks}**",
        f"- `tenure` values of those blank rows: **{r.total_charges_blank_tenure_values}**",
        "- Rule: blanks correspond to brand-new (`tenure == 0`) customers and are",
        "  imputed to **0.0** (no charges accrued yet). Documented and unit-tested.",
        "",
        "## Duplicate rows",
        "",
        f"- Fully identical rows: **{r.duplicate_rows}**",
        "- Decision: **reported, not dropped.** OpenML omits `customerID`, so an",
        "  identical feature+label row is plausibly a distinct real customer;",
        "  dropping would risk discarding genuine data.",
        "",
        "## Missing values after cleaning",
        "",
        f"- Total missing: **{sum(r.missing_after.values())}** (target: 0)",
        "",
        "## Collinearity flag (tenure / MonthlyCharges / TotalCharges)",
        "",
        f"- corr(tenure, TotalCharges) = **{r.collinearity.get('tenure~TotalCharges')}**",
        f"- corr(MonthlyCharges, TotalCharges) = **{r.collinearity.get('MonthlyCharges~TotalCharges')}**",
        f"- corr(tenure, MonthlyCharges) = **{r.collinearity.get('tenure~MonthlyCharges')}**",
        "",
        "`TotalCharges` is approximately `tenure x MonthlyCharges`. It is retained",
        "but this near-collinearity is surfaced here, in the EDA, and in the SHAP",
        "driver discussion so feature importances are not over-interpreted.",
        "",
        "## Cardinality (categorical columns)",
        "",
    ]
    for col, n in sorted(r.cardinality.items()):
        lines.append(f"- `{col}`: {n}")
    lines += ["", "## Assertions passed", ""]
    for a in r.assertions_passed:
        lines.append(f"- {a}")
    lines.append(churn_note)
    (REPORTS_DIR / "data_quality.md").write_text("\n".join(lines))
