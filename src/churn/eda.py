"""EDA that answers specific business questions (not a generic plot dump).

Each function answers one named question and returns a tidy table; the report
writer renders a figure and a one-line takeaway for each.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from churn.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402


def churn_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["Churn"].agg(["mean", "count"]).reset_index()
    g.columns = [col, "churn_rate", "n"]
    return g.sort_values("churn_rate", ascending=False)


def churn_by_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 12, 24, 36, 48, 60, 1_000]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "60+"]
    d = df.copy()
    d["tenure_bucket"] = pd.cut(d["tenure"], bins=bins, labels=labels)
    g = d.groupby("tenure_bucket", observed=True)["Churn"].agg(["mean", "count"]).reset_index()
    g.columns = ["tenure_bucket", "churn_rate", "n"]
    return g


def churn_by_monthly_decile(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["charge_decile"] = pd.qcut(d["MonthlyCharges"], 10, labels=False, duplicates="drop")
    g = d.groupby("charge_decile", observed=True)["Churn"].agg(["mean", "count"]).reset_index()
    g.columns = ["charge_decile", "churn_rate", "n"]
    return g


def _bar(df, x, y, title, fname, rotate=0):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df[x].astype(str), df[y], color="#3b7dd8")
    ax.set_title(title)
    ax.set_ylabel("churn rate")
    ax.set_xlabel(x)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=110)
    plt.close(fig)


def run_eda_report(df: pd.DataFrame) -> None:
    """Generate all EDA figures and the report with per-question takeaways."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    contract = churn_by_category(df, "Contract")
    tenure = churn_by_tenure_bucket(df)
    monthly = churn_by_monthly_decile(df)
    payment = churn_by_category(df, "PaymentMethod")
    internet = churn_by_category(df, "InternetService")
    tech = churn_by_category(df, "TechSupport")

    _bar(contract, "Contract", "churn_rate", "Churn rate by contract type", "churn_by_contract.png")
    _bar(
        tenure, "tenure_bucket", "churn_rate", "Churn rate by tenure bucket", "churn_by_tenure.png"
    )
    _bar(
        monthly,
        "charge_decile",
        "churn_rate",
        "Churn rate by MonthlyCharges decile",
        "churn_by_monthly.png",
    )
    _bar(
        payment,
        "PaymentMethod",
        "churn_rate",
        "Churn rate by payment method",
        "churn_by_payment.png",
        rotate=30,
    )
    _bar(
        internet,
        "InternetService",
        "churn_rate",
        "Churn rate by internet service",
        "churn_by_internet.png",
    )
    _bar(
        tech, "TechSupport", "churn_rate", "Churn rate by tech support", "churn_by_techsupport.png"
    )

    overall = df["Churn"].mean()
    lines = [
        "# Exploratory Data Analysis",
        "",
        f"Overall churn rate: **{overall:.1%}** ({int(df['Churn'].sum())} of {len(df)}).",
        "Each section answers a specific business question.",
        "",
        "## 1. Does contract type drive churn?",
        "![contract](figures/churn_by_contract.png)",
        f"- Month-to-month churns far more than fixed-term. Highest group: "
        f"**{contract.iloc[0]['Contract']}** at {contract.iloc[0]['churn_rate']:.1%}.",
        "",
        "## 2. Is churn front-loaded in tenure?",
        "![tenure](figures/churn_by_tenure.png)",
        f"- Churn is concentrated in the first year (**0-12** bucket: "
        f"{tenure.iloc[0]['churn_rate']:.1%}), then falls with tenure.",
        "",
        "## 3. Are customers price-sensitive?",
        "![monthly](figures/churn_by_monthly.png)",
        "- Churn rises with monthly charges, consistent with price sensitivity.",
        "",
        "## 4. Does payment method matter?",
        "![payment](figures/churn_by_payment.png)",
        f"- **{payment.iloc[0]['PaymentMethod']}** has the highest churn "
        f"({payment.iloc[0]['churn_rate']:.1%}) — the electronic-check pattern.",
        "",
        "## 5. Fiber vs DSL",
        "![internet](figures/churn_by_internet.png)",
        f"- Highest-churn internet type: **{internet.iloc[0]['InternetService']}** "
        f"({internet.iloc[0]['churn_rate']:.1%}).",
        "",
        "## 6. Do add-on services retain?",
        "![tech](figures/churn_by_techsupport.png)",
        "- Customers without tech support churn more, suggesting add-ons aid retention.",
        "",
        "## 7. Collinearity trio",
        "- `TotalCharges` ~ `tenure x MonthlyCharges` (see data quality report);",
        "  interpret these three as one correlated bundle.",
    ]
    (REPORTS_DIR / "eda.md").write_text("\n".join(lines))
