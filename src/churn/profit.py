"""Profit layer: per-customer CLV, expected-value targeting, threshold
optimization, and sensitivity analysis.

IMPORTANT: every number produced here is a **simulated expected value under
assumed economics** (margin, discount, lifetime, offer cost, success rate). It
is NOT measured profit. The assumptions live in ``config.ProfitAssumptions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import DEFAULT_ASSUMPTIONS, ProfitAssumptions


def customer_clv(monthly_charges: np.ndarray, a: ProfitAssumptions) -> np.ndarray:
    """Discounted per-customer CLV (option a).

    CLV = sum_{m=1..L} (MonthlyCharges * gross_margin) / (1 + r_m)^m

    where L = expected_lifetime_months and r_m = monthly discount rate. Vectorized
    over an array of monthly charges.
    """
    monthly_charges = np.asarray(monthly_charges, dtype=float)
    r = a.monthly_discount_rate
    months = np.arange(1, a.expected_lifetime_months + 1)
    discount_factors = 1.0 / np.power(1.0 + r, months)
    # Sum of discount factors is the annuity factor; CLV scales linearly in margin*charge.
    annuity = float(discount_factors.sum())
    return monthly_charges * a.gross_margin * annuity


def expected_value_if_targeted(
    p_churn: np.ndarray, clv: np.ndarray, a: ProfitAssumptions
) -> np.ndarray:
    """EV of sending a retention offer to each customer.

    value_if_target = p_churn * offer_success_rate * CLV - offer_cost

    Rationale: the offer only "saves" value for customers who would otherwise
    churn (prob p_churn) and who the offer successfully retains
    (offer_success_rate). The offer cost is paid regardless.
    """
    p_churn = np.asarray(p_churn, dtype=float)
    clv = np.asarray(clv, dtype=float)
    return p_churn * a.offer_success_rate * clv - a.offer_cost


def total_profit_at_threshold(
    p_churn: np.ndarray,
    clv: np.ndarray,
    threshold: float,
    a: ProfitAssumptions,
) -> float:
    """Total simulated EV if we target everyone with p_churn >= threshold."""
    ev = expected_value_if_targeted(p_churn, clv, a)
    target = p_churn >= threshold
    return float(ev[target].sum())


def optimize_threshold(
    p_churn: np.ndarray,
    clv: np.ndarray,
    a: ProfitAssumptions = DEFAULT_ASSUMPTIONS,
    grid: np.ndarray | None = None,
) -> dict:
    """Find the threshold maximizing total simulated EV.

    Should be called with OUT-OF-FOLD training predictions for threshold
    selection; the resulting threshold is then applied once on the test set.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    profits = np.array([total_profit_at_threshold(p_churn, clv, t, a) for t in grid])
    best_i = int(np.argmax(profits))
    best_t = float(grid[best_i])
    n_targeted = int((p_churn >= best_t).sum())
    return {
        "best_threshold": best_t,
        "best_profit": float(profits[best_i]),
        "n_targeted": n_targeted,
        "grid": grid,
        "profits": profits,
    }


def baseline_profits(
    p_churn: np.ndarray, clv: np.ndarray, a: ProfitAssumptions
) -> dict[str, float]:
    """Profit of naive policies, for context on the model's uplift.

    - ``target_none``: 0 by definition (no offers sent).
    - ``target_all``: send an offer to every customer (threshold 0).
    These bracket the model-driven policy and quantify the value of targeting.
    """
    ev = expected_value_if_targeted(p_churn, clv, a)
    return {
        "target_none": 0.0,
        "target_all": float(ev.sum()),
    }


def profit_confidence_interval(
    p_churn: np.ndarray,
    clv: np.ndarray,
    threshold: float,
    a: ProfitAssumptions,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for total simulated profit at a FIXED threshold.

    Customers are resampled with replacement; profit is recomputed each draw.
    This expresses sampling uncertainty in the headline number (the threshold is
    held fixed — it was chosen on training OOF, so we do not re-select it here).
    """
    p_churn = np.asarray(p_churn, dtype=float)
    clv = np.asarray(clv, dtype=float)
    ev = expected_value_if_targeted(p_churn, clv, a)
    targeted = p_churn >= threshold
    per_customer = np.where(targeted, ev, 0.0)

    rng = np.random.default_rng(seed)
    n = len(per_customer)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = per_customer[idx].sum()
    point = float(per_customer.sum())
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"profit": point, "ci_lo": lo, "ci_hi": hi}


def threshold_robustness(
    p_churn: np.ndarray,
    clv: np.ndarray,
    best_threshold: float,
    a: ProfitAssumptions,
    delta: float = 0.1,
    n_points: int = 9,
) -> pd.DataFrame:
    """Profit across a neighborhood of the chosen threshold.

    Shows whether the optimum is a sharp spike (fragile) or a broad plateau
    (robust). Returns a tidy DataFrame of (threshold, profit, pct_of_best).
    """
    lo = max(0.0, best_threshold - delta)
    hi = min(1.0, best_threshold + delta)
    grid = np.linspace(lo, hi, n_points)
    best = total_profit_at_threshold(p_churn, clv, best_threshold, a)
    rows = []
    for t in grid:
        prof = total_profit_at_threshold(p_churn, clv, float(t), a)
        rows.append(
            {
                "threshold": round(float(t), 4),
                "profit": round(prof, 2),
                "pct_of_best": round(100.0 * prof / best, 1) if best else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_grid(
    p_churn: np.ndarray,
    monthly_charges: np.ndarray,
    base: ProfitAssumptions,
    offer_costs: np.ndarray,
    success_rates: np.ndarray,
) -> pd.DataFrame:
    """Break-even surface: optimal profit over (offer_cost, success_rate).

    For each combination, recompute CLV (unaffected by these two levers but
    recomputed for clarity), re-optimize the threshold, and record the best
    profit and whether targeting is net-positive. Returns a long DataFrame.
    """
    rows = []
    for oc in offer_costs:
        for sr in success_rates:
            a = base.with_overrides(offer_cost=float(oc), offer_success_rate=float(sr))
            clv = customer_clv(monthly_charges, a)
            res = optimize_threshold(p_churn, clv, a)
            rows.append(
                {
                    "offer_cost": float(oc),
                    "success_rate": float(sr),
                    "best_threshold": res["best_threshold"],
                    "best_profit": res["best_profit"],
                    "n_targeted": res["n_targeted"],
                    "targeting_pays_off": res["best_profit"] > 0,
                }
            )
    return pd.DataFrame(rows)
