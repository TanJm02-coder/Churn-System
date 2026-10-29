"""Profit-logic tests: EV monotonicity, correct optimum, CLV discounting."""

from __future__ import annotations

import numpy as np

from churn.config import ProfitAssumptions
from churn.profit import (
    baseline_profits,
    customer_clv,
    expected_value_if_targeted,
    optimize_threshold,
    profit_confidence_interval,
    threshold_robustness,
    total_profit_at_threshold,
)


def test_clv_discounting_matches_closed_form():
    a = ProfitAssumptions(gross_margin=0.5, expected_lifetime_months=3, annual_discount_rate=0.12)
    r = a.monthly_discount_rate  # 0.01
    monthly = np.array([100.0])
    expected = 100.0 * 0.5 * sum(1.0 / (1.0 + r) ** m for m in (1, 2, 3))
    got = customer_clv(monthly, a)[0]
    assert np.isclose(got, expected)


def test_clv_zero_discount_is_simple_sum():
    a = ProfitAssumptions(gross_margin=1.0, expected_lifetime_months=12, annual_discount_rate=0.0)
    got = customer_clv(np.array([10.0]), a)[0]
    assert np.isclose(got, 10.0 * 12)  # no discount -> 12 * margin * charge


def test_ev_is_monotonic_increasing_in_churn_prob():
    a = ProfitAssumptions()
    clv = np.full(5, 1000.0)
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ev = expected_value_if_targeted(p, clv, a)
    assert np.all(np.diff(ev) > 0), "EV should strictly increase with churn prob"


def test_optimizer_finds_true_max_on_toy_case():
    """Two customers: one clearly worth targeting, one clearly not.

    High-p/high-CLV customer has positive EV; low-p customer has negative EV.
    The optimal threshold must target only the first.
    """
    a = ProfitAssumptions(offer_cost=50.0, offer_success_rate=0.5)
    # Cust 1: EV = 0.9*0.5*2000 - 50 = 850 (>0). Cust 2: 0.02*0.5*2000 - 50 = -30 (<0).
    p = np.array([0.9, 0.02])
    clv = np.array([2000.0, 2000.0])
    ev = expected_value_if_targeted(p, clv, a)
    assert ev[0] > 0 and ev[1] < 0  # sanity on the toy setup

    res = optimize_threshold(p, clv, a, grid=np.linspace(0, 1, 101))
    t = res["best_threshold"]
    targeted = p >= t
    assert targeted[0] and not targeted[1]
    assert np.isclose(res["best_profit"], ev[0])


def test_total_profit_respects_threshold():
    a = ProfitAssumptions()
    p = np.array([0.2, 0.8])
    clv = np.array([1000.0, 1000.0])
    # At threshold 0 everyone targeted; at 1.0 nobody targeted -> 0 profit.
    assert total_profit_at_threshold(p, clv, 1.01, a) == 0.0
    all_targeted = total_profit_at_threshold(p, clv, 0.0, a)
    assert np.isclose(all_targeted, expected_value_if_targeted(p, clv, a).sum())


def test_baseline_target_none_is_zero_and_all_sums_ev():
    a = ProfitAssumptions()
    p = np.array([0.2, 0.8, 0.5])
    clv = np.array([1000.0, 1000.0, 1000.0])
    base = baseline_profits(p, clv, a)
    assert base["target_none"] == 0.0
    assert np.isclose(base["target_all"], expected_value_if_targeted(p, clv, a).sum())


def test_optimized_profit_dominates_naive_baselines():
    """The profit-maximizing threshold must be >= both naive policies."""
    a = ProfitAssumptions(offer_cost=50.0, offer_success_rate=0.5)
    p = np.array([0.95, 0.9, 0.05, 0.02])
    clv = np.full(4, 2000.0)
    res = optimize_threshold(p, clv, a)
    base = baseline_profits(p, clv, a)
    assert res["best_profit"] >= base["target_none"]
    assert res["best_profit"] >= base["target_all"]


def test_profit_ci_brackets_point_estimate():
    a = ProfitAssumptions()
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    clv = rng.uniform(500, 3000, 500)
    t = 0.4
    ci = profit_confidence_interval(p, clv, t, a, n_boot=500)
    assert ci["ci_lo"] <= ci["profit"] <= ci["ci_hi"]
    # CI should be reproducible given the fixed seed.
    ci2 = profit_confidence_interval(p, clv, t, a, n_boot=500)
    assert np.isclose(ci["ci_lo"], ci2["ci_lo"]) and np.isclose(ci["ci_hi"], ci2["ci_hi"])


def test_threshold_robustness_peaks_at_chosen_threshold():
    a = ProfitAssumptions()
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 400)
    clv = rng.uniform(500, 3000, 400)
    res = optimize_threshold(p, clv, a)
    t = res["best_threshold"]
    rob = threshold_robustness(p, clv, t, a)
    # pct_of_best is 100 at the chosen threshold and <= 100 everywhere in range.
    assert rob["pct_of_best"].max() <= 100.0 + 1e-6
    at_best = rob.loc[np.isclose(rob["threshold"], round(t, 4)), "pct_of_best"]
    if len(at_best):
        assert np.isclose(at_best.iloc[0], 100.0, atol=0.5)
