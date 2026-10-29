# Customer Churn Prediction & Retention System — Requirements

**Status:** DRAFT — awaiting approval. No implementation code will be written until this document is signed off.

**Author:** Kiro (for review)
**Last updated:** 2026-10-28

---

## 0. Purpose & business questions

A portfolio-grade system that answers three business questions, in priority order:

1. **Which customers are likely to leave?** (ranking / probability)
2. **Which retention offers are worth their cost?** (per-customer expected-value targeting)
3. **What decision threshold maximizes expected profit?** (headline metric)

Model quality metrics (AUC etc.) serve these questions; the **profit-maximizing threshold is the headline result**, not AUC.

---

## 1. Data

- **Source:** OpenML dataset **ID 42178** ("Telco-Customer-Churn"), fetched via `sklearn.datasets.fetch_openml` / OpenML API. Public, no auth.
- **Reproducibility:** `scripts/download_data.py` writes the raw CSV to `data/raw/telco_churn.csv`. The script is idempotent (skips if present unless `--force`), records the OpenML version and a SHA-256 checksum of the downloaded file to `data/raw/DATA_MANIFEST.json`.
- `data/raw/` and `data/processed/` are **gitignored**. The manifest is committed so the exact data version is reproducible.
- **Target:** `Churn` (Yes/No → 1/0). Churn rate ≈ 26.5% — **class imbalance** handled explicitly (stratified CV, PR-AUC emphasis, `class_weight` / `scale_pos_weight`).

### 1.1 Known data facts to handle
- `TotalCharges` is typed as string and contains ~11 blank values corresponding to `tenure == 0` (brand-new customers). These are **imputed to 0.0** (a 0-tenure customer has accrued no total charges) and flagged in the cleaning report; the imputation rule is documented and tested.
- `customerID` is dropped (identifier, no signal, leakage-neutral).
- `SeniorCitizen` is already 0/1 integer; others are categorical strings.

---

## 2. Data cleaning (with documented quality checks)

Module: `src/churn/data.py`. Produces `data/processed/telco_clean.parquet` and a **cleaning report** (`reports/data_quality.md`) covering:

- Row/column counts before and after each step.
- Missing-value audit (pre/post), with the `TotalCharges` blank-string handling called out.
- Dtype coercions (`TotalCharges` → float; binary Yes/No → category; target → int).
- Duplicate-row check (expect 0; assert and report).
- Cardinality report for every categorical column.
- Range/sanity assertions (`tenure` ≥ 0, `MonthlyCharges` > 0, target ∈ {0,1}).
- **Collinearity flag:** report correlation among `tenure`, `MonthlyCharges`, `TotalCharges` and note that `TotalCharges ≈ tenure × MonthlyCharges`. Documented as an interpretability caveat; `TotalCharges` is retained but the collinearity is surfaced in EDA and the SHAP discussion.

All cleaning invariants are enforced by assertions and covered by unit tests.

---

## 3. EDA — answering specific business questions

Module: `src/churn/eda.py`, output to `reports/eda.md` + figures in `reports/figures/`. Each figure answers a named question, **not** a generic dump:

1. **Does contract type drive churn?** Churn rate by `Contract` (month-to-month vs 1/2-year).
2. **Is churn front-loaded in tenure?** Churn rate by tenure bucket (0–12, 13–24, …).
3. **Are customers price-sensitive?** Churn rate vs `MonthlyCharges` deciles.
4. **Does payment method matter?** Churn rate by `PaymentMethod` (electronic check hypothesis).
5. **Do add-on services retain?** Churn rate by `OnlineSecurity`, `TechSupport`, etc.
6. **Fiber vs DSL:** churn by `InternetService`.
7. **The collinearity trio:** `tenure`/`MonthlyCharges`/`TotalCharges` relationships, visualized.

Each gets a one-line plain-language takeaway in the report.

---

## 4. Modeling

Module: `src/churn/models.py`, `src/churn/pipeline.py`.

### 4.0 Train/test split protocol (held-out evaluation)
- At the very start of modeling, a **stratified 80/20 train/test split** (seed 42) is created. The split indices are persisted so every downstream step uses the same partition.
- **The test set is never used for model selection, calibration, or threshold choice.** All of those decisions are made with **stratified CV on the training set only**:
  - Model selection: by CV PR-AUC on training folds.
  - Calibration: fit on training data only (within CV / a held-out slice of train).
  - **Profit-maximizing threshold: chosen from out-of-fold (OOF) predictions on the training set.**
- **Final metrics, calibration curves, and the profit result are reported once on the test set** — these are the headline numbers.
- A unit test (§8) **fails if any test-set row appears in any training or calibration step** (index-overlap check between the fitted data and the held-out test indices).

### 4.1 Feature handling
- **`gender` is EXCLUDED from model features** (sensitive attribute; the system decides who receives offers). Retained in a side frame for the fairness audit only. **This decision is documented** in the README and in code comments.
- Preprocessing via a `ColumnTransformer` inside a `Pipeline`:
  - Numeric: median impute (defensive) + standardize (for LogReg; trees indifferent but harmless or skipped per-model).
  - Categorical: one-hot encode (`handle_unknown="ignore"`).
- **All preprocessing fit on training folds only** — never on the full dataset. This is enforced by the pipeline structure and by an explicit leakage test (§8).

### 4.2 Models
1. **Logistic Regression** — baseline, `class_weight="balanced"`.
2. **Random Forest** — `class_weight="balanced"`.
3. **XGBoost** — `scale_pos_weight` set from class ratio. (`xgboost` added as a dependency.)
4. **Calibrated best model** — the best performer (by CV PR-AUC) wrapped in `CalibratedClassifierCV` (isotonic if enough data, else sigmoid/Platt); calibration compared before/after.

### 4.3 Evaluation
- **Stratified K-fold CV** (k=5, seed pinned) for model comparison.
- Metrics per model: **ROC-AUC, PR-AUC**, **recall at precision ∈ {0.3, 0.5, 0.7}**, and **calibration curves** (reliability diagram + Brier score).
- Results table written to `reports/model_comparison.md`.
- **CV scores on the training set are used for model selection; the test-set numbers are the headline.** This distinction is stated in the README.
- A single global seed (`RANDOM_STATE = 42`) threaded through all stochastic components.

---

## 5. Interpretability

Module: `src/churn/interpret.py`.
- **SHAP** values on the best tree model (TreeExplainer). Global importance (beeswarm/bar) + a few local explanations.
- **Plain-language driver summary** in `reports/drivers.md`: top churn drivers in business terms, explicitly noting the `TotalCharges`/`tenure` collinearity caveat so importances aren't over-read.

---

## 6. Segmentation

Module: `src/churn/segmentation.py`.
- **K-means** (default) with k chosen via silhouette + elbow; features standardized. (Hierarchical offered as an alternative in a helper, k-means is the committed path.)
- **Segment profiles**: size, churn rate, avg `MonthlyCharges`/`tenure`, dominant contract/payment, written description per segment.
- **Model performance by segment**: PR-AUC / recall per segment, surfacing where the model is weak. **Report sample size `n` per segment, and bootstrap 95% confidence intervals** for the per-segment metrics where practical (small segments flagged as low-confidence). Written to `reports/segments.md`.

---

## 7. Profit layer (the headline)

Module: `src/churn/profit.py`.

### 7.1 Per-customer CLV (option a)
```
remaining_lifetime_months = expected_lifetime_months (assumption, default 24)
                            # optionally scaled by contract type
monthly_margin             = MonthlyCharges * gross_margin      # default gross_margin = 0.30
CLV (discounted)           = Σ_{m=1..L} monthly_margin / (1 + monthly_discount)^m
```
- **Documented assumptions & defaults** (all adjustable in the app):
  - **Gross margin:** `0.30` (telco service margins commonly cited 20–40%; 30% is a defensible midpoint — documented as an assumption, not a fact).
  - **Expected remaining lifetime:** `24` months (horizon cap; documented).
  - **Annual discount rate:** `0.10` → monthly ≈ `0.10/12`. Standard mid-range cost-of-capital assumption, documented.
  - **Offer cost:** `$50` per targeted customer.
  - **Offer success rate:** `0.30` (prob. a targeted churner is retained).

### 7.2 Expected-value decision
For each customer with churn probability `p`:
```
value_if_target   = p * offer_success_rate * CLV  -  offer_cost
value_if_no_target = 0
target when value_if_target > 0
```
A **threshold optimizer** sweeps thresholds; the optimal threshold is chosen from **out-of-fold predictions on the training set**, then the resulting profit is reported **once on the test set** (the headline number).

> **Framing:** the profit headline is a **simulated expected value under assumed economics** (margin, discount, lifetime, offer cost, success rate) — it is **not measured profit**. This is stated wherever the number appears, including the README and the app.

### 7.3 Sensitivity analysis
- Vary offer cost, success rate, margin, and discount rate; produce a **break-even surface / heatmap** showing the region where targeting is net-positive, and how the optimal threshold moves. Written to `reports/profit.md`.

---

## 8. Fairness check (report & discuss only — no auto-mitigation)

Module: `src/churn/fairness.py`.
- Groups audited: **`gender`, `SeniorCitizen`, `Partner`, `Dependents`**.
- Metrics per group: **FPR, FNR, recall, selection rate** (at the chosen profit threshold), plus **group sample size `n`** and **bootstrap 95% confidence intervals** on the rates where practical (small groups flagged as low-confidence).
- Written discussion in `reports/fairness.md`: interpret disparities, note that `gender` is excluded as a feature yet still audited, and discuss that mitigation is deliberately **not** applied (and why that's itself a tradeoff).
- **Proxy caveat:** explicitly note that **excluding `gender` does not remove proxy effects** — the model can still encode group information indirectly through correlated features, and `SeniorCitizen`, `Partner`, and `Dependents` remain as features and may act as proxies.

---

## 9. Streamlit app (fuller version)

Module: `app/streamlit_app.py`. Three views, **no upload-a-customer scoring**:
1. **Profit & threshold:** sliders for offer cost, success rate, gross margin, discount rate, expected lifetime → live optimal threshold, expected profit, # targeted, confusion outcomes.
2. **Segment explorer:** pick a segment → profile + model performance.
3. **SHAP view:** global driver importance + the plain-language summary.

Loads pre-trained artifacts from `models/` (no training in the app).

---

## 10. Engineering, testing, docs

- **Tooling:** `uv` for env/deps, `ruff` for format+lint (committed `pyproject.toml` config). Python 3.11.
- **Project layout:**
  ```
  src/churn/{data,eda,pipeline,models,interpret,segmentation,profit,fairness,config}.py
  scripts/{download_data,run_pipeline}.py
  app/streamlit_app.py
  tests/
  reports/  models/  data/
  ```
- **Tests (`pytest`), must include:**
  - **Leakage test (preprocessing):** asserts preprocessing/transformers are fit on train folds only — a deliberately-constructed check that fails if any transformer is fit on full data (e.g. verifying the scaler's learned stats differ from full-data stats / that the pipeline is cloned per fold).
  - **Leakage test (test-set isolation):** fails if any **test-set row index** appears in the data used for any training or calibration step (index-overlap check against the held-out test indices).
  - **Preprocessing checks:** one-hot handles unseen categories; no NaNs leak through; `gender` is absent from the model feature matrix.
  - **Cleaning invariants:** `TotalCharges` blanks → 0.0; dtypes; no duplicates; target ∈ {0,1}.
  - **Profit logic:** EV formula monotonic in `p`; threshold optimizer picks the true max on a toy case; CLV discounting correct.
- **README** with: overview, reproducible run instructions, results summary, and a prominent **Limitations** section stating:
  - **This is a snapshot dataset with no temporal split.** The model identifies **correlates of churn**, not a validated **forward-looking predictor**; some features may be measured at/after the churn decision.
  - **Methodology:** CV scores on the training set drive model selection, calibration, and threshold choice; the **test-set numbers are the headline**. The held-out test set is untouched until final evaluation.
  - The profit headline is a **simulated expected value under assumed economics, not measured profit**.
  - `TotalCharges`/`tenure`/`MonthlyCharges` collinearity caveat.
  - CLV/margin/discount/success-rate numbers are **assumptions**, not measured values.
  - `gender` excluded as a feature by design; fairness is audited, not mitigated. **Excluding `gender` does not remove proxy effects** through correlated features including `SeniorCitizen`, `Partner`, and `Dependents`.
  - Per-segment and per-group metrics report sample sizes and bootstrap CIs; small groups are low-confidence.

---

## 11. Delivery & review protocol

- Requirements shown here **before building** (this document).
- After building, an **audit pass** is run and findings reported **before any further changes**.
- Work delivered on a branch / PR for review (not committed to `main`).

---

## 12. Out of scope (explicit)

- Upload-a-customer live scoring in the app.
- Automatic fairness mitigation.
- Real-time / streaming inference, deployment infra.
- Hyperparameter search beyond light, documented tuning.
