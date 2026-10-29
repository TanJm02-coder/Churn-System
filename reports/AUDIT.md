# Churn System — Correctness & Integrity Audit

**Date:** 2026-10-28
**Scope:** Read-only audit of `src/churn/`, `scripts/`, `app/`, `tests/` against `REQUIREMENTS.md`.
**Focus (priority order):** data leakage, profit logic, fairness, requirements coverage, test quality.

---

## Summary verdict

The pipeline's **train/test split isolation is genuinely sound** — the held-out test set is created once, indices persisted, and model selection / calibration / threshold choice all run on the training set via CV. That core requirement (§4.0, §7.2) holds.

However, there is **one critical correctness bug** that undermines the headline profit number: the profit threshold is optimized on **uncalibrated** out-of-fold probabilities but then applied to **calibrated** test probabilities. These two probability scales are different, so the threshold is not optimal for the distribution it is applied to. There are also several **moderate** issues (SHAP computed on the held-out test set; calibration folds not seeded consistently with the rest of the pipeline; the CV comparison reported as per-fold when it is actually pooled OOF; a latent fragility in how `gender` is reattached for the fairness audit) and, most importantly, the **leakage tests are weak** — `test_test_rows_absent_from_training_data` and `test_preprocessing_fit_on_train_fold_only` can pass even if real leakage existed in the production path, because they exercise constructs the production code does not use.

Counts: **1 critical**, **5 moderate**, **4 minor**.

---

## CRITICAL

### C1. Threshold chosen on uncalibrated OOF probabilities, applied to calibrated test probabilities
**Files:** `scripts/run_pipeline.py:52-66`, `src/churn/models.py:66-96` (`cv_compare`), `src/churn/models.py:116-126` (`fit_calibrated`)

The flow is:

```
oof_best = oof[best]                      # from cv_compare -> PLAIN pipeline, UNCALIBRATED proba
thr_res  = optimize_threshold(oof_best, clv_train, ...)   # threshold picked on uncalibrated scale
best_threshold = thr_res["best_threshold"]

proba_test = calibrated.predict_proba(x_test)[:, 1]        # CALIBRATED scale
test_profit = total_profit_at_threshold(proba_test, clv_test, best_threshold, ...)
```

`cv_compare` (`models.py:84-90`) builds `build_model_pipeline(est, ...)` with **no calibration** and produces OOF probabilities from that plain estimator. For `random_forest`/`xgboost` with `class_weight`/`scale_pos_weight`, the raw `predict_proba` is systematically miscalibrated and shifted toward the minority class. The threshold `optimize_threshold` finds is tuned to *that* shifted distribution.

At test time the probabilities come from `CalibratedClassifierCV` (isotonic), which remaps those raw scores onto a calibrated scale. A probability of, say, 0.55 on the uncalibrated OOF scale does not correspond to 0.55 on the calibrated test scale. Applying the uncalibrated-derived threshold to calibrated probabilities targets a **different, non-optimal** customer set, so the headline "profit-maximizing threshold" (REQUIREMENTS §0, §7.2) is not actually profit-maximizing for the numbers reported.

This is not a test-set-leakage bug (isolation is preserved); it is a **threshold-transfer correctness bug** that directly corrupts the headline metric. The requirement (§7.2) says the threshold is chosen from OOF train predictions and *the resulting profit is reported on test* — that only makes sense if the OOF predictions and the test predictions are on the same scale.

**Fix direction:** produce OOF probabilities from the **calibrated** estimator (e.g. `cross_val_predict` over `CalibratedClassifierCV(pipe, cv=...)`, or an outer CV that calibrates within each fold), and optimize the threshold on *those*. Then the threshold and the test probabilities live on the same scale.

---

## MODERATE

### M1. SHAP interpretability is computed on the held-out TEST set
**Files:** `scripts/run_pipeline.py:74-79`, `src/churn/interpret.py:22-55`

```
imp, _, _ = interpret.shap_importance(tree_pipe, x_test)
```

The tree model is fit on train (`fit_plain(best_tree, x_train, y_train)`) but SHAP importances and the `drivers.md` report are generated from `x_test`. This is not label leakage into the model, but it does use the held-out set for an analysis product. REQUIREMENTS §4.0 states "The test set is never used for model selection, calibration, or threshold choice" — SHAP is none of those, so this is not a strict violation, but §5 does not require test data and using it muddies the "test set untouched until final evaluation" claim in the README/§10. The `plain_language_summary` text even labels it "(test-set sample)". Prefer computing SHAP on `x_train` (or a train subsample) to keep the test set purely for the single headline evaluation.

### M2. Calibration uses `cv=CV_FOLDS` (plain int) — not stratified, shuffle semantics unclear
**File:** `src/churn/models.py:123-125`

`CalibratedClassifierCV(pipe, method=method, cv=CV_FOLDS)` passes `cv=5` as an integer. For a classifier target this uses `StratifiedKFold` but **without `shuffle=True` and without a pinned `random_state`**, unlike every other CV in the project (`cv_compare` uses `StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)`). REQUIREMENTS §4.3 requires "a single global seed threaded through all stochastic components." The calibration folds are not seeded consistently with the rest of the pipeline, so calibration (and therefore the test probabilities and the profit headline) is not fully reproducible/consistent with the stated protocol.

### M3. Model-comparison report mislabels CV scores as a clean selection basis while PR-AUC is computed on OOF predictions pooled across folds
**Files:** `src/churn/models.py:91-96`, `src/churn/metrics.py:31-41`

`cv_compare` computes `core_scores(y_train, proba)` where `proba` is the pooled OOF vector from `cross_val_predict`. Pooling OOF predictions and scoring once is a legitimate approach, but it is **not** the same as the per-fold-averaged "Stratified K-fold CV (k=5)" metric table that REQUIREMENTS §4.3 implies ("Metrics per model ... Stratified K-fold CV"). There are no per-fold means or variances, so the report cannot show fold-level stability. This is a methodology gap, not a bug, but the `model_comparison.md` wording ("stratified 5-fold CV") overstates what is actually reported (single pooled-OOF score). Recommend either per-fold scoring with mean±std, or relabel as "pooled out-of-fold" scoring.

### M4. `select_best` can pick a non-tree model, but `plot_calibration` / headline always compare uncalibrated-vs-calibrated of the *selected* model while SHAP silently switches models
**Files:** `scripts/run_pipeline.py:48-79`, `src/churn/models.py:100-113`

If `logreg` wins CV PR-AUC, `best = "logreg"`, and `select_best_tree` picks a *different* model (`xgboost`/`random_forest`) for SHAP. The drivers report then explains a model that is **not** the one producing the churn probabilities, thresholds, profit, or fairness results. REQUIREMENTS §5 says "SHAP values on the best tree model," so computing SHAP on a tree is intended — but the report should state prominently that the explained model may differ from the scoring model. As written, `drivers.md` reads as if it explains the deployed scorer. Flag the mismatch in the report.

### M5. Fairness audit uses a calibrated-probability threshold but gender reattachment relies on sort-order coupling that is fragile
**Files:** `scripts/run_pipeline.py:97-126` (`x_test_with_gender`), `src/churn/split.py:45-70`

The reattachment logic is **correct as currently written**, but only because of a non-obvious coupling: `stratified_split` sorts `test_idx` ascending (`split.py:54`) and builds `df_test = df.iloc[test_idx].reset_index(drop=True)`, while `x_test_with_gender` does `df.iloc[test_idx].reset_index(drop=True)` with `test_idx = load_split_indices()["test_idx"]` (also ascending, as persisted). Because both paths use the same ascending index order and both `reset_index(drop=True)`, row *i* of `x_test` aligns with row *i* of the reattached gender. 

The fragility: this alignment holds **only** while `test_idx` is persisted and reloaded in the same order and neither side reorders rows. There is no assertion tying `x_test` rows to `test_idx` positions (e.g. no check that `len(gender) == len(x_test)` or that a stable key matches). If anyone later changes the sort, drops the `.sort()`, or filters rows, gender will silently misalign and the fairness audit will attribute error rates to the wrong individuals with no error raised. Add a hard alignment assertion (ideally join on a stable key rather than positional order). Confidence that it is correct today: **confirmed**; the risk is latent fragility, not a present miscomputation.

---

## MINOR

### m1. `SeniorCitizen` is numeric, so it is standardized and one-hot logic never sees it — fairness grouping still works but EDA/segmentation treat it as a numeric feature
**Files:** `src/churn/pipeline.py:21-27`, `src/churn/data.py:106`

`clean` keeps `SeniorCitizen` as int, so `split_feature_types` routes it to the numeric branch (scaled for logreg, passed through for trees). That is defensible, but it means `SeniorCitizen` is the only binary attribute treated as continuous. Grouping in fairness (`df_test["SeniorCitizen"] == value`) still works because the raw 0/1 values are preserved in `x_test`. No correctness bug; worth a one-line note.

### m2. Group error-rate math is correct; selection rate denominator is whole-group (intended) — document it
**File:** `src/churn/fairness.py:21-35`

`recall`, `fpr`, `fnr` use the standard confusion-cell denominators and are **correct**. `selection_rate = (tp + fp) / len(y_true)` uses the full group size as denominator (fraction of the group that receives an offer), which is the right definition for "who gets targeted." Just note that `fnr = 1 - recall` is reported redundantly (both derived from the same cells); harmless but duplicative.

### m3. Bootstrap CIs for recall/selection resample the rate vector, not (y_true, y_pred) jointly
**Files:** `src/churn/fairness.py:60-72`, `src/churn/metrics.py:44-72`

Recall CI resamples `recall_vals = (yp[pos_mask] == 1)` — i.e. it bootstraps only over in-group positives, which is a reasonable binomial CI for recall. Fine. Just note the CI is a simple percentile bootstrap on a Bernoulli vector; for very small positive counts it will be wide/degenerate (the code already flags small groups). Acceptable per §8 ("where practical").

### m4. CLV formula and EV logic are correct and match REQUIREMENTS §7.1/§7.2
**Files:** `src/churn/profit.py:17-55`, verified against `tests/test_profit.py`

`customer_clv` computes `Σ_{m=1..L} margin*charge / (1+r)^m` with `r = annual/12` — matches §7.1 exactly, including the monthly conversion. `expected_value_if_targeted = p*success*CLV - cost` matches §7.2. `optimize_threshold` sweeps a 201-point grid and takes `argmax` of total EV — correct. The monotonicity test (`test_ev_is_monotonic_increasing_in_churn_prob`) holds because EV is linear-increasing in `p`. **No issue** — recorded here as a positive confirmation. (Note the §7.1 "optionally scaled by contract type" is not implemented; it is explicitly optional, so not a gap.)

---

## Test-quality findings (priority item 5)

### T1. `test_test_rows_absent_from_training_data` cannot fail on the real leakage mode (weak test)
**File:** `tests/test_leakage.py:66-82`

The test overwrites `MonthlyCharges` with `np.arange(len)+1` to fingerprint rows, then checks `x_train` and `x_test` fingerprints are disjoint. But `stratified_split` **constructs** train and test by `df.iloc[train_idx]` / `df.iloc[test_idx]` from disjoint index arrays — so the two frames are disjoint *by construction* regardless of any downstream leakage. The requirement (§4.0, §8) asks for an **index-overlap check between the data actually fed to each fit/calibration step and the held-out test indices**. This test never inspects what `cv_compare`, `fit_calibrated`, or `optimize_threshold` were given; it only re-checks that the splitter splits. It would still pass if, e.g., `fit_calibrated` were accidentally called on the full frame. **Confidence: confirmed.** It tests the splitter, not test-set isolation of the fitting steps.

### T2. `test_preprocessing_fit_on_train_fold_only` tests `build_preprocessor` directly, not the production CV path (weak test)
**File:** `tests/test_leakage.py:24-57`

The test manually fits a `build_preprocessor` on a fold vs on full data and asserts the learned means differ. That proves `StandardScaler.fit` depends on its input (a tautology about sklearn) — it does **not** prove that the production pipeline (`cross_val_predict(build_model_pipeline(...))` in `cv_compare`) refits preprocessing per fold. If someone replaced the pipeline with a pre-fitted transformer, or fit the preprocessor on full `x_train` before CV, this test would still pass. The §8 requirement ("verifying ... the pipeline is cloned per fold") is not actually exercised against `cv_compare`. **Confidence: confirmed.** Strengthen by asserting on the clone behavior of the real `cv_compare`/pipeline (e.g. check that a fold estimator's fitted scaler stats differ from a full-data fit *through the same pipeline construction the pipeline uses*).

### T3. No test asserts the threshold is derived from OOF (not test), nor the scale-match in C1
**Files:** `tests/test_leakage.py`, `tests/test_profit.py`

There is no test covering REQUIREMENTS §7.2's core claim — that the threshold comes from OOF training predictions and is validly transferable to the test set. The C1 bug (uncalibrated→calibrated mismatch) is exactly the kind of error a test should catch and none does. Recommend a test that the OOF probabilities feeding `optimize_threshold` come from the *same estimator family/scale* as the probabilities used at test time.

### T4. Split-coverage test uses `repeat(30)`, masking the stratification edge the real data has
**File:** `tests/test_leakage.py:59-64`

`test_train_test_indices_are_disjoint` replicates the 6-row toy frame 30× so stratification is trivial. It confirms disjointness and full coverage (good), but does not test stratification proportions or the `.sort()` behavior that `x_test_with_gender` depends on (see M5). Minor.

**Positive:** `test_profit.py` is solid — it checks the closed-form CLV, the zero-discount case, EV monotonicity, and that the optimizer targets the right customer on a toy case with a known analytic optimum. `tests/test_cleaning.py` and `tests/test_preprocessing.py` correctly cover the §1.1/§4.1 invariants (TotalCharges blank→0.0, target domain, no-NaN, gender absent, unseen-category handling).

---

## Requirements coverage cross-check (priority item 4)

| Req | Status | Note |
|-----|--------|------|
| §1 data download/manifest/idempotent | Not audited in depth | `scripts/download_data.py` present; not read line-by-line this pass. |
| §1.1 TotalCharges blanks→0.0, SeniorCitizen int | **Met** | `data.py:85-96`, tested. |
| §2 cleaning report (counts, dtypes, cardinality, collinearity, assertions) | **Met** | `data.py` + `data_quality.md`. Duplicates reported-not-dropped, documented. |
| §3 EDA named-question figures | Partially verified | Figures exist in `reports/figures/`; `eda.py` not read this pass. |
| §4.0 split isolation; threshold from OOF | **Partially — see C1** | Isolation met; threshold scale mismatch breaks the "resulting profit on test" intent. |
| §4.1 gender excluded; preprocessing fit on folds | **Met** | `split.make_feature_frame`, pipeline structure; tested (weakly, see T2). |
| §4.2 LogReg/RF/XGB + calibrated best | **Met** | `models.build_estimators`, `fit_calibrated`. |
| §4.3 ROC/PR-AUC, recall@precision, calibration, single seed | **Partially** | Metrics present; seed not threaded into `CalibratedClassifierCV` (M2); CV reported as pooled OOF not per-fold (M3). |
| §5 SHAP on best tree + collinearity caveat | **Met w/ caveat** | Computed on **test** set (M1); explained model may differ from scorer (M4). |
| §6 k-means, silhouette, per-segment perf + CIs | **Met** | `segmentation.py`; CIs and low-confidence flags present. |
| §7.1 CLV discounting | **Met** | `profit.customer_clv` matches formula. |
| §7.2 EV targeting + threshold optimizer | **Met (logic) / see C1 (application)** | Formula correct; cross-scale application flawed. |
| §7.3 sensitivity surface | **Met** | `sensitivity_grid` over cost×success; `profit.md`. |
| §8 fairness groups, FPR/FNR/recall/selection, n + CIs, proxy caveat | **Met** | `fairness.audit`; reattachment correct-but-fragile (M5). |
| §9 Streamlit 3 views, no upload scoring, loads artifacts | Not audited this pass | `app/streamlit_app.py` not read. |
| §10 tests (leakage ×2, preprocessing, cleaning, profit) | **Present but leakage tests weak** | See T1/T2/T3. |
| §10 README limitations section | Not audited this pass | `README.md` not read. |

---

## Recommended priority of fixes
1. **C1** — recompute OOF on the calibrated estimator (or calibrate within folds) so the threshold and test probabilities share one scale. This is the only issue that corrupts the headline number.
2. **T1/T2/T3** — rewrite the two leakage tests to exercise the real `cv_compare`/calibration path, and add a threshold-scale test. Current tests give false assurance.
3. **M2** — seed `CalibratedClassifierCV`'s CV (`StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)`).
4. **M1/M4** — move SHAP to train data and label the explained-vs-scoring model clearly.
5. **M5** — add a hard alignment assertion in `x_test_with_gender`.

*Read-only audit; no source files were modified.*

---

## Resolution log (2026-10-28, post-audit fixes)

All audit findings were addressed. Summary of changes:

- **C1 (critical) — FIXED.** `cv_compare` now produces out-of-fold probabilities
  from the **calibrated** estimator (`build_calibrated_pipeline`), so the profit
  threshold (chosen on OOF) and the calibrated test predictions share one scale.
  Headline moved: threshold 0.405 → **0.350**, simulated test EV $17,858 →
  **$18,253** (+$395 / +2.2%); recall@0.5 0.794 → 0.805, recall@0.7 0.406 →
  0.414; metrics otherwise ~flat.
- **M1 — FIXED.** SHAP is now computed on a **training-set** sample, not the test
  set; `drivers.md` wording updated.
- **M2 — FIXED.** Calibration and all CV use a shared seeded splitter
  (`make_cv`, `StratifiedKFold(shuffle=True, random_state=42)`).
- **M3 — FIXED.** `model_comparison.md` now reports both pooled-OOF scores and
  **per-fold mean ± std**; methodology text corrected.
- **M4 — FIXED.** `drivers.md` prominently states SHAP explains the best tree
  model (`random_forest`), which differs from the deployed scorer (`logreg`).
- **M5 — FIXED.** `x_test_with_gender` now hard-asserts positional alignment on
  shared columns (`MonthlyCharges`, `tenure`, `SeniorCitizen`) before attaching
  `gender`.
- **Minors — FIXED.** OpenML ARFF single-quotes stripped from categorical values
  in cleaning (reports now read `Fiber optic`, not `'Fiber optic'`); fairness
  report documents `SeniorCitizen`-as-numeric, the redundant `fnr`, and CI
  caveats.

### Leakage tests rewritten (T1/T2/T3)
- **T1** — a row-recording spy wraps the **real** calibrated pipeline inside
  `cross_val_predict`; asserts no held-out test-row id reaches any `fit` call.
- **T2** — reconstructs the production pipeline and fits per fold on real fold
  indices; asserts scaler stats differ across folds (true per-fold refit).
- **T3** — asserts OOF and refit-calibrated probabilities share one scale.
- **Injection proofs** (`tests/test_leakage_detects_injected.py`) deliberately
  introduce each leak and assert the guard fires. Suite: **23 passed**, 0
  skipped; `ruff check`/`format` clean.

---

## Second-pass finding (2026-10-28, pre-push mutation + duplicate checks)

- **Mutation tests.** Reverting C1 (uncalibrated OOF) initially did NOT fail the
  original T3 (its mean-gap tolerance was too loose for the selected logreg).
  T3 was strengthened to an element-wise comparison against calibrated vs
  uncalibrated reference OOF; it now fails decisively under the C1 mutation.
  Injecting a test row into the training frame fails the T1 row-spy as expected.
- **Duplicate rows straddling the split — FIXED.** An audit check found that
  **7 of 20 duplicate groups straddled the train/test split** (7 test rows had an
  identical training twin), a minor memorized-row leakage. (An earlier ad-hoc
  manual check wrongly reported 0 — the groupby-based test is authoritative.)
  Fix: `deduplicate_for_split` drops 22 exact-duplicate rows (~0.3%) **before**
  splitting; cleaning still reports-not-drops on the full data. Enforced by
  `tests/test_split_duplicates.py`. Post-dedup headline (honest, un-inflated):
  threshold 0.350 → **0.365**, EV $18,253 → **$17,601**, recall@0.5 0.805 →
  0.772, recall@0.7 0.414 → 0.470. Suite: **25 passed**.

---

## Enhancements (2026-10-28, "improve it further")

Additive analytics strengthening the profit layer (no behavior change to models,
threshold, or headline EV):

- **Bootstrap 95% CI on the headline profit.** `profit_confidence_interval`
  resamples test customers (threshold fixed) → test EV $17,601, **95% CI
  [$15,643, $19,714]**. Surfaced in `headline.json`, `profit.md`, and the app.
- **Naive-policy baselines + uplift.** `baseline_profits` quantifies that
  *targeting everyone* loses ≈ **−$16,471** (offer cost wasted on low-risk
  customers), so the model-driven policy is ≈ **+$34,072** better — a concrete
  demonstration of targeting value.
- **Threshold robustness table.** `threshold_robustness` shows profit stays
  within ~6% of best across ±0.1 on the threshold (broad plateau, not a
  knife-edge).
- **Makefile** with setup/data/pipeline/test/lint/format/app targets.
- **4 new profit tests** (baselines, optimizer-dominates-baselines, CI brackets
  point estimate + reproducible, robustness peaks at the chosen threshold).

Suite: **29 passed**; `ruff` clean. Headline unchanged (threshold 0.365, EV
$17,601) — these are reporting/analytics additions, not model changes.
