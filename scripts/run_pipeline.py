"""End-to-end pipeline: clean -> split -> EDA -> CV select -> calibrate ->
test eval -> SHAP -> segments -> profit -> fairness. Saves artifacts for the app.

Run:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from churn import eda, evaluate, fairness, interpret, models, profit, segmentation
from churn.config import (
    DEFAULT_ASSUMPTIONS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from churn.data import deduplicate_for_split as dedup_for_split
from churn.data import run_cleaning_report
from churn.metrics import core_scores
from churn.split import stratified_split


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean + quality report.
    print("[1/8] cleaning ...")
    df, _ = run_cleaning_report()

    # 2. EDA.
    print("[2/8] EDA ...")
    eda.run_eda_report(df)

    # 3. Deduplicate exact rows BEFORE splitting to prevent memorized-row
    #    leakage (identical rows straddling train/test), then stratified split.
    print("[3/8] dedup + split ...")
    df_model, n_dropped = dedup_for_split(df)
    print(
        f"      dropped {n_dropped} exact-duplicate rows before split "
        f"({len(df)} -> {len(df_model)})"
    )
    x_train, x_test, y_train, y_test = stratified_split(df_model, persist=True)

    # 4. CV model comparison on TRAIN only. OOF probabilities come from the
    #    CALIBRATED estimator so they share one scale with the final test model.
    print("[4/8] cross-validating models (train only, calibrated OOF) ...")
    cv_scores, oof_cal, per_fold = models.cv_compare(x_train, y_train, calibrated=True)
    fold_summary = models.per_fold_summary(per_fold)
    best = models.select_best(cv_scores)
    print(f"      selected: {best}")

    # 5. Fit the final calibrated model (same construction as the OOF estimator),
    #    plus a plain uncalibrated fit for the before/after calibration curve.
    print("[5/8] fitting final calibrated model (train only) ...")
    calibrated = models.fit_calibrated(best, x_train, y_train)
    plain = models.fit_plain(best, x_train, y_train)

    # 6. Threshold from CALIBRATED OOF train predictions; final eval ONCE on test.
    #    Both the OOF probabilities and the test probabilities are on the
    #    calibrated scale, so the threshold is validly transferable (fixes C1).
    print("[6/8] threshold from calibrated OOF (train), final eval on test ...")
    oof_best = oof_cal[best]
    clv_train = profit.customer_clv(x_train["MonthlyCharges"].to_numpy(), DEFAULT_ASSUMPTIONS)
    thr_res = profit.optimize_threshold(oof_best, clv_train, DEFAULT_ASSUMPTIONS)
    best_threshold = thr_res["best_threshold"]

    proba_test = calibrated.predict_proba(x_test)[:, 1]
    proba_test_plain = plain.predict_proba(x_test)[:, 1]
    test_scores = core_scores(y_test.to_numpy(), proba_test)
    briers = evaluate.plot_calibration(
        y_test.to_numpy(),
        {"uncalibrated": proba_test_plain, "calibrated": proba_test},
    )
    evaluate.write_model_comparison(cv_scores, fold_summary, best, test_scores, briers)

    # Profit on the TEST set using the train-chosen threshold (headline).
    clv_test = profit.customer_clv(x_test["MonthlyCharges"].to_numpy(), DEFAULT_ASSUMPTIONS)
    test_profit = profit.total_profit_at_threshold(
        proba_test, clv_test, best_threshold, DEFAULT_ASSUMPTIONS
    )
    profit_ci, profit_baselines = _write_profit_report(
        thr_res, best_threshold, test_profit, proba_test, clv_test, x_test
    )

    # 7. SHAP on the best TREE model (TreeExplainer needs a tree ensemble).
    #    Computed on TRAIN data so the held-out test set stays untouched (M1).
    print("[7/8] SHAP interpretability (on training data) ...")
    best_tree = models.select_best_tree(cv_scores)
    tree_pipe = models.fit_plain(best_tree, x_train, y_train)
    print(f"      SHAP on best tree model: {best_tree}")
    # Subsample train for tractable SHAP on larger data.
    shap_sample = x_train.sample(n=min(len(x_train), 1000), random_state=RANDOM_STATE)
    imp, _, _ = interpret.shap_importance(tree_pipe, shap_sample)
    (REPORTS_DIR / "drivers.md").write_text(
        interpret.plain_language_summary(imp, explained_model=best_tree, scoring_model=best)
    )
    imp.to_csv(MODELS_DIR / "shap_importance.csv", index=False)

    # 8. Segmentation + per-segment performance; fairness audit.
    print("[8/8] segmentation + fairness ...")
    km, train_labels, seg_preproc = segmentation.fit_segments(x_train)
    test_feats = seg_preproc.transform(x_test)
    test_labels = km.predict(test_feats)
    seg_profiles = segmentation.profile_segments(
        x_test.assign(Churn=y_test.to_numpy()), test_labels
    )
    seg_perf = segmentation.performance_by_segment(y_test.to_numpy(), proba_test, test_labels)
    _write_segments_report(seg_profiles, seg_perf)

    y_pred_test = (proba_test >= best_threshold).astype(int)
    fair = fairness.audit(x_test_with_gender(df_model, x_test), y_test.to_numpy(), y_pred_test)
    _write_fairness_report(fair, best_threshold)

    # Persist artifacts for the app.
    joblib.dump(
        {
            "calibrated_model": calibrated,
            "plain_model": plain,
            "best_name": best,
            "best_threshold": best_threshold,
            "kmeans": km,
            "seg_preprocessor": seg_preproc,
            "tree_model": tree_pipe,
            "best_tree_name": best_tree,
        },
        MODELS_DIR / "artifacts.joblib",
    )
    pd.DataFrame(
        {
            "proba": proba_test,
            "y": y_test.to_numpy(),
            "MonthlyCharges": x_test["MonthlyCharges"].to_numpy(),
            "segment": test_labels,
        }
    ).to_parquet(MODELS_DIR / "test_scored.parquet")
    (MODELS_DIR / "headline.json").write_text(
        json.dumps(
            {
                "best_model": best,
                "best_threshold": best_threshold,
                "test_profit_simulated_ev": test_profit,
                "test_profit_ci95": [profit_ci["ci_lo"], profit_ci["ci_hi"]],
                "profit_vs_target_all": profit_baselines["target_all"],
                "test_scores": test_scores,
                "brier": briers,
            },
            indent=2,
        )
    )
    print(
        f"\nDONE. Headline (simulated EV): threshold={best_threshold:.3f} "
        f"profit=${test_profit:,.0f} on test set."
    )


def x_test_with_gender(df: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    """Reattach the excluded ``gender`` column to the test rows for the audit.

    ``SeniorCitizen``/``Partner``/``Dependents`` are already present in x_test
    (they remain model features); only ``gender`` was dropped, so only it is
    reattached, aligned by the persisted test indices.
    """
    from churn.split import load_split_indices

    test_idx = load_split_indices()["test_idx"]
    reattached = df.iloc[test_idx].reset_index(drop=True)
    out = x_test.reset_index(drop=True).copy()

    # Hard alignment assertions (M5): the reattached rows must match x_test row
    # for row. We verify length and cross-check columns that exist in BOTH frames
    # (e.g. MonthlyCharges, tenure) to confirm positional alignment before
    # attaching the gender column that x_test does not carry.
    assert len(reattached) == len(out), "gender reattachment length mismatch"
    for shared in ("MonthlyCharges", "tenure", "SeniorCitizen"):
        if shared in out.columns and shared in reattached.columns:
            assert (out[shared].to_numpy() == reattached[shared].to_numpy()).all(), (
                f"gender reattachment misaligned on '{shared}' — row order differs"
            )

    out["gender"] = reattached["gender"].to_numpy()
    return out


def _write_profit_report(thr_res, threshold, test_profit, proba_test, clv_test, x_test):
    a = DEFAULT_ASSUMPTIONS
    sens = profit.sensitivity_grid(
        proba_test,
        x_test["MonthlyCharges"].to_numpy(),
        a,
        offer_costs=np.array([25, 50, 75, 100, 150]),
        success_rates=np.array([0.15, 0.30, 0.45, 0.60]),
    )
    # Headline uncertainty, naive-policy baselines, and threshold robustness.
    ci = profit.profit_confidence_interval(proba_test, clv_test, threshold, a)
    baselines = profit.baseline_profits(proba_test, clv_test, a)
    robustness = profit.threshold_robustness(proba_test, clv_test, threshold, a)
    uplift_vs_all = test_profit - baselines["target_all"]

    lines = [
        "# Profit Layer (Simulated Expected Value)",
        "",
        "> **All figures here are a simulated expected value under assumed",
        "> economics, not measured profit.**",
        "",
        "## Assumptions (defaults; adjustable in the app)",
        f"- Offer cost: ${a.offer_cost:.0f}",
        f"- Offer success rate: {a.offer_success_rate:.0%}",
        f"- Gross margin: {a.gross_margin:.0%}",
        f"- Expected lifetime cap: {a.expected_lifetime_months} months",
        f"- Annual discount rate: {a.annual_discount_rate:.0%}",
        "",
        "## Threshold selection",
        "- Threshold chosen on **out-of-fold training predictions**, then applied",
        "  **once on the held-out test set**.",
        f"- Chosen threshold: **{threshold:.3f}**",
        f"- Customers targeted on test set: **{int((proba_test >= threshold).sum())}** "
        f"of {len(proba_test)}",
        f"- **Headline simulated EV on test set: ${test_profit:,.0f}**",
        f"- **Bootstrap 95% CI: [${ci['ci_lo']:,.0f}, ${ci['ci_hi']:,.0f}]** "
        f"(1000 resamples of test customers; threshold held fixed)",
        "",
        "## Model-targeting uplift vs naive policies",
        "",
        f"- Target **no one**: ${baselines['target_none']:,.0f}",
        f"- Target **everyone**: ${baselines['target_all']:,.0f}",
        f"- Model-chosen threshold: **${test_profit:,.0f}** "
        f"(**+${uplift_vs_all:,.0f}** vs targeting everyone)",
        "",
        "The model-driven policy beats both naive extremes: it avoids wasting",
        "offer cost on low-risk customers (unlike target-everyone) while still",
        "capturing most retainable value (unlike target-no-one).",
        "",
        "## Threshold robustness (neighborhood of the optimum)",
        "",
        robustness.to_markdown(index=False),
        "",
        "A broad plateau near 100% of best profit indicates the optimum is robust,",
        "not a fragile knife-edge; small threshold mis-estimates cost little.",
        "",
        "## Sensitivity (offer cost x success rate)",
        "",
        sens.round(2).to_markdown(index=False),
        "",
        "Rows with `targeting_pays_off = False` mark the regime where the program",
        "is net-negative under those assumptions.",
    ]
    (REPORTS_DIR / "profit.md").write_text("\n".join(lines))
    return ci, baselines


def _write_segments_report(profiles, perf):
    lines = [
        "# Customer Segments (k-means)",
        "",
        "k selected by silhouette score. Per-segment model performance includes",
        "sample size and bootstrap 95% CIs; `low_confidence = True` flags segments",
        "with few positives where PR-AUC is unstable.",
        "",
        "## Segment profiles",
        "",
        profiles.to_markdown(index=False),
        "",
        "## Model performance by segment (test set)",
        "",
        perf.to_markdown(index=False),
    ]
    (REPORTS_DIR / "segments.md").write_text("\n".join(lines))


def _write_fairness_report(fair, threshold):
    lines = [
        "# Fairness Audit (Report & Discuss Only)",
        "",
        f"Error rates computed at the profit threshold **{threshold:.3f}** on the",
        "test set. **No mitigation is applied** — this is a reporting and",
        "discussion layer.",
        "",
        "Each rate includes group sample size `n` and bootstrap 95% CIs for recall",
        "and selection rate; narrow groups are low-confidence.",
        "",
        fair.to_markdown(index=False),
        "",
        "## Discussion",
        "",
        "- **`gender` is excluded from the model's features by design**, yet it is",
        "  audited here. Excluding it does **not** remove proxy effects: the model",
        "  can still encode group information through correlated features, and",
        "  `SeniorCitizen`, `Partner`, and `Dependents` remain model features and",
        "  may act as proxies.",
        "- Compare FPR/FNR/recall across groups. A higher FNR for a group means",
        "  that group's at-risk customers are under-targeted for retention offers;",
        "  a higher selection rate means more offers (and cost) directed at it.",
        "- Mitigation (e.g. group-specific thresholds) is deliberately omitted: it",
        "  introduces its own fairness tradeoffs and is out of scope by agreement.",
        "",
        "## Notes on method",
        "",
        "- `SeniorCitizen` is the one sensitive attribute stored as 0/1 and is",
        "  treated as a numeric feature by the model; grouping here uses its raw",
        "  0/1 values, so the audit is unaffected.",
        "- `fnr` is reported alongside `recall` for readability even though",
        "  `fnr = 1 - recall`; both are derived from the same confusion cells.",
        "- `n` is the group size; recall and selection-rate CIs are percentile",
        "  bootstraps and will be wide for small groups (flagged as low-confidence",
        "  in the segmentation report).",
    ]
    (REPORTS_DIR / "fairness.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
