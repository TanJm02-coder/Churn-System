"""Streamlit app: profit & threshold, segment explorer, and SHAP view.

Loads pre-trained artifacts from ``models/`` (no training in the app). Run:

    streamlit run app/streamlit_app.py

Assumptions are adjustable; every profit figure is a SIMULATED expected value
under those assumptions, not measured profit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Make the src/ package importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import DEFAULT_ASSUMPTIONS, MODELS_DIR  # noqa: E402
from churn.profit import (  # noqa: E402
    baseline_profits,
    customer_clv,
    optimize_threshold,
    profit_confidence_interval,
    total_profit_at_threshold,
)

st.set_page_config(page_title="Churn Retention", layout="wide")


@st.cache_resource
def load_artifacts():
    art = joblib.load(MODELS_DIR / "artifacts.joblib")
    scored = pd.read_parquet(MODELS_DIR / "test_scored.parquet")
    imp_path = MODELS_DIR / "shap_importance.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else None
    return art, scored, imp


def main() -> None:
    if not (MODELS_DIR / "artifacts.joblib").exists():
        st.error("Artifacts not found. Run `python scripts/run_pipeline.py` first.")
        return

    art, scored, imp = load_artifacts()
    st.title("Customer Churn Prediction & Retention")
    st.caption(
        "Profit figures are a **simulated expected value under assumed economics**, "
        "not measured profit. Snapshot dataset — the model finds correlates of churn, "
        "not a validated forward-looking predictor."
    )

    tab_profit, tab_segments, tab_shap = st.tabs(
        ["Profit & Threshold", "Segment Explorer", "SHAP Drivers"]
    )

    with tab_profit:
        _profit_view(scored)
    with tab_segments:
        _segment_view(scored)
    with tab_shap:
        _shap_view(imp)


def _sidebar_assumptions():
    st.sidebar.header("Economic assumptions")
    d = DEFAULT_ASSUMPTIONS
    offer_cost = st.sidebar.slider("Offer cost ($)", 0, 300, int(d.offer_cost), 5)
    success = st.sidebar.slider("Offer success rate", 0.0, 1.0, d.offer_success_rate, 0.05)
    margin = st.sidebar.slider("Gross margin", 0.0, 1.0, d.gross_margin, 0.05)
    lifetime = st.sidebar.slider("Expected lifetime (months)", 1, 60, d.expected_lifetime_months, 1)
    disc = st.sidebar.slider("Annual discount rate", 0.0, 0.5, d.annual_discount_rate, 0.01)
    return d.with_overrides(
        offer_cost=float(offer_cost),
        offer_success_rate=float(success),
        gross_margin=float(margin),
        expected_lifetime_months=int(lifetime),
        annual_discount_rate=float(disc),
    )


def _profit_view(scored: pd.DataFrame):
    a = _sidebar_assumptions()
    p = scored["proba"].to_numpy()
    clv = customer_clv(scored["MonthlyCharges"].to_numpy(), a)

    res = optimize_threshold(p, clv, a)
    t = res["best_threshold"]
    targeted = p >= t

    ci = profit_confidence_interval(p, clv, t, a, n_boot=500)
    base = baseline_profits(p, clv, a)

    c1, c2, c3 = st.columns(3)
    c1.metric("Profit-max threshold", f"{t:.3f}")
    c2.metric(
        "Simulated EV (test set)",
        f"${res['best_profit']:,.0f}",
        help=f"Bootstrap 95% CI: ${ci['ci_lo']:,.0f} to ${ci['ci_hi']:,.0f}",
    )
    c3.metric(
        "Customers targeted",
        f"{int(targeted.sum())} / {len(p)}",
        help=f"vs target-everyone (${base['target_all']:,.0f})",
    )
    st.caption(
        f"Bootstrap 95% CI on simulated EV: "
        f"**\\${ci['ci_lo']:,.0f} – \\${ci['ci_hi']:,.0f}** · "
        f"uplift vs targeting everyone: **\\${res['best_profit'] - base['target_all']:,.0f}**"
    )

    st.subheader("Expected value vs threshold")
    curve = pd.DataFrame({"threshold": res["grid"], "simulated_ev": res["profits"]})
    st.line_chart(curve.set_index("threshold"))

    st.subheader("Outcome breakdown at the chosen threshold")
    y = scored["y"].to_numpy()
    tp = int(((targeted) & (y == 1)).sum())
    fp = int(((targeted) & (y == 0)).sum())
    fn = int(((~targeted) & (y == 1)).sum())
    tn = int(((~targeted) & (y == 0)).sum())
    st.table(
        pd.DataFrame(
            {
                "Targeted churner (TP)": [tp],
                "Targeted non-churner (FP)": [fp],
                "Missed churner (FN)": [fn],
                "Correctly skipped (TN)": [tn],
            }
        )
    )
    st.caption(
        f"Manual what-if: profit at threshold 0.5 would be "
        f"${total_profit_at_threshold(p, clv, 0.5, a):,.0f}."
    )


def _segment_view(scored: pd.DataFrame):
    if "segment" not in scored.columns:
        st.info("No segment data available.")
        return
    segs = sorted(scored["segment"].unique().tolist())
    seg = st.selectbox("Select a segment", segs)
    g = scored[scored["segment"] == seg]

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers (test)", f"{len(g)}")
    c2.metric("Actual churn rate", f"{g['y'].mean():.1%}")
    c3.metric("Avg monthly charge", f"${g['MonthlyCharges'].mean():.0f}")

    st.subheader("All segments: size and churn rate")
    summary = (
        scored.groupby("segment")
        .agg(n=("y", "size"), churn_rate=("y", "mean"), avg_monthly=("MonthlyCharges", "mean"))
        .round(3)
    )
    st.dataframe(summary)
    st.caption("Per-segment model performance with CIs is in reports/segments.md.")


def _shap_view(imp: pd.DataFrame | None):
    st.subheader("Global churn drivers (mean |SHAP|)")
    if imp is None:
        st.info("SHAP importances not found. Re-run the pipeline.")
        return
    top = imp.head(15).set_index("feature")["mean_abs_shap"]
    st.bar_chart(top)
    st.markdown(
        "**Caveat:** `TotalCharges` ≈ `tenure × MonthlyCharges`. These three move "
        "together, so SHAP importance can be split among them — read them as one "
        "correlated bundle. Full write-up in `reports/drivers.md`."
    )


if __name__ == "__main__":
    main()
