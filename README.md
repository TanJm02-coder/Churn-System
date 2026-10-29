# Customer Churn Prediction & Retention System

A portfolio-grade, end-to-end system on the IBM Telco Customer Churn dataset that
answers three business questions:

1. **Which customers are likely to leave?** (calibrated churn probabilities)
2. **Which retention offers are worth their cost?** (per-customer expected value)
3. **What threshold maximizes expected profit?** (the headline decision)

> ⚠️ **Read the [Limitations](#limitations) section before trusting any number here.**
> The profit figures are a **simulated expected value under assumed economics**,
> and the dataset is a **snapshot with no temporal split** — the model finds
> *correlates* of churn, not a validated *forward-looking predictor*.

---

## Results at a glance

Selected model: **logistic regression** (highest cross-validated PR-AUC on the
training set). SHAP interpretability runs on the best **tree** model (random
forest). Headline numbers are measured **once on a held-out 20% test set**:

| Metric (test set)        | Value   |
|--------------------------|---------|
| ROC-AUC                  | 0.840   |
| PR-AUC                   | 0.633   |
| Recall @ precision 0.30  | 0.995   |
| Recall @ precision 0.50  | 0.772   |
| Recall @ precision 0.70  | 0.470   |
| Brier (uncalibrated)     | 0.167   |
| Brier (calibrated)       | 0.138   |
| **Profit-max threshold** | **0.365** |
| **Simulated EV (test)**  | **≈ $17,600** (95% CI ≈ [$15,600, $19,700]) |

(Numbers are post-deduplication — see Limitations item 9.) Calibration improved
the Brier score (0.167 → 0.138).

**Why targeting matters:** under the default assumptions, targeting *everyone*
would lose money (≈ −$16,500, since a \$50 offer is wasted on thousands of
low-risk customers). The model-driven threshold turns that into ≈ +$17,600 — an
uplift of ≈ \$34,000 over the target-everyone policy, and the optimum sits on a
broad plateau (profit stays within ~6% of best across ±0.1 on the threshold), so
it is not a fragile knife-edge. See `reports/profit.md` for the CI, the
naive-policy comparison, the robustness table, and the full sensitivity surface. The profit threshold is
chosen on **calibrated** out-of-fold training predictions and applied to the
calibrated test predictions (same probability scale). Full tables and figures
are regenerated into `reports/` by the pipeline.

---

## Methodology (how selection vs. headline works)

- A **stratified 80/20 train/test split** (seed 42) is made once; indices are
  persisted to `models/split_indices.json`.
- **The test set is never used for model selection, calibration, or threshold
  choice.** Those decisions use **stratified 5-fold CV on the training set
  only** (seeded). The profit threshold is chosen from **out-of-fold (OOF)
  training predictions produced by the calibrated estimator**, so it transfers
  to the calibrated test predictions on the same probability scale.
- **CV scores drive model selection; the test-set numbers are the headline**,
  reported exactly once.

---

## Project structure

```
src/churn/
  config.py          seeds, paths, economic assumptions (single source of truth)
  data.py            cleaning + documented quality checks
  split.py           stratified split; gender excluded from features here
  pipeline.py        ColumnTransformer + model pipeline (leakage-safe)
  metrics.py         recall@precision, PR-AUC, bootstrap CIs
  models.py          LogReg / RF / XGBoost, CV compare, calibration
  evaluate.py        calibration curves + model-comparison report
  interpret.py       SHAP importance + plain-language driver summary
  segmentation.py    k-means + profiles + per-segment performance (n + CIs)
  profit.py          CLV, EV targeting, threshold optimizer, sensitivity
  fairness.py        group error rates with n + bootstrap CIs
  eda.py             business-question EDA figures + report
scripts/
  download_data.py   reproducible OpenML download + checksum manifest
  run_pipeline.py    end-to-end orchestration; writes reports/ and models/
app/streamlit_app.py profit/threshold, segment explorer, SHAP views
tests/               cleaning, preprocessing, leakage, profit tests
reports/             generated Markdown reports + figures
```

---

## Setup & reproduce

Requires Python 3.11+. Uses [`uv`](https://github.com/astral-sh/uv).

**Quick path (Makefile):**

```bash
make setup      # create venv + install
make data       # download OpenML dataset
make pipeline   # run full pipeline -> reports/ + models/
make test       # run the test suite
make lint       # ruff check + format --check
make app        # launch the Streamlit app
```

**Or explicitly:**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/download_data.py      # -> data/raw/ + DATA_MANIFEST.json
python scripts/run_pipeline.py       # -> reports/ + models/ artifacts
python -m pytest                     # run the test suite
ruff check . && ruff format --check . # lint + format

streamlit run app/streamlit_app.py   # interactive app (loads pre-trained artifacts)
```

`data/raw/` and `data/processed/` are gitignored; `DATA_MANIFEST.json` (OpenML
version + SHA-256) is committed so the exact data version is reproducible.

> Note: on some sandboxes set `MPLCONFIGDIR=/tmp/mpl` so matplotlib has a
> writable cache.

---

## The profit layer (simulated expected value)

Per-customer discounted CLV (option a):

```
CLV = Σ_{m=1..L}  (MonthlyCharges × gross_margin) / (1 + r_month)^m
```

**Documented assumptions** (defaults; all adjustable in the app). None are
measured from the data:

| Assumption            | Default | Notes                                           |
|-----------------------|---------|-------------------------------------------------|
| Offer cost            | $50     | per targeted customer                           |
| Offer success rate    | 30%     | P(targeted churner retained)                    |
| Gross margin          | 30%     | telco midpoint (commonly cited 20–40%)          |
| Expected lifetime cap | 24 mo   | horizon cap → finite, conservative CLV          |
| Annual discount rate  | 10%     | cost-of-capital; converted to monthly           |

Targeting rule: `EV_target = p_churn × success_rate × CLV − offer_cost`; target
when `EV_target > 0`. The optimizer sweeps thresholds; `reports/profit.md`
includes a sensitivity table over offer cost × success rate showing where
targeting is net-positive.

---

## Interpretability & segments

- **SHAP** (TreeExplainer on the best tree model) → `reports/drivers.md` with a
  plain-language driver summary.
- **k-means segmentation** (k by silhouette) → `reports/segments.md`. The
  high-risk segment (month-to-month, electronic check, low tenure) churns ~44%
  and is where the model performs best (PR-AUC ≈ 0.68); a small segment is
  flagged `low_confidence` (few positives). Per-segment metrics report sample
  size and bootstrap 95% CIs.

---

## Fairness

`reports/fairness.md` compares **FPR, FNR, recall, and selection rate** across
**gender, SeniorCitizen, Partner, Dependents**, each with sample size and
bootstrap 95% CIs. **Report & discuss only — no mitigation.** Notable: senior
citizens show a higher false-positive and selection rate; customers with
dependents show a high false-negative rate (their at-risk members are
under-targeted).

---

## Limitations

**Please read these before using any result.**

1. **Snapshot, no temporal split.** This is a cross-sectional snapshot, not a
   time series. There is no way to prove the absence of look-ahead leakage, and
   some features (e.g. `Contract`, `TotalCharges`) may be measured at or after
   the churn decision. **The model identifies *correlates* of churn — it is not
   a validated *forward-looking* predictor.** Treat it as a prioritization aid,
   not a forecast.
2. **Selection vs. headline.** CV scores on the training set drive model
   selection, calibration, and threshold choice; the **test-set numbers are the
   headline**. A single train/test split gives one estimate — repeated nested CV
   would tighten confidence.
3. **Simulated profit, not measured.** The profit headline is a **simulated
   expected value under assumed economics** (margin, discount, lifetime, offer
   cost, success rate). It is **not measured profit** and depends entirely on
   assumptions that should be validated against real business data.
4. **Collinearity.** `TotalCharges ≈ tenure × MonthlyCharges`
   (corr(tenure, TotalCharges) ≈ 0.83). These three move together, so feature
   importances can be split or shifted among them; read them as one bundle.
5. **Gender excluded, but proxies remain.** `gender` is excluded from the model's
   features by design (the system decides who receives offers), yet audited for
   fairness. **Excluding it does not remove proxy effects**: the model can encode
   group information through correlated features, and `SeniorCitizen`, `Partner`,
   and `Dependents` remain features and may act as proxies.
6. **Fairness is reported, not mitigated.** Mitigation (e.g. group-specific
   thresholds) is deliberately out of scope; it introduces its own tradeoffs.
7. **Small-group uncertainty.** Per-segment and per-group metrics include sample
   sizes and bootstrap CIs; small groups are flagged low-confidence and should
   not drive decisions on their own.
8. **Offer-success rate is assumed uniform.** In reality it likely varies by
   customer and offer type; the model treats it as a single lever.
9. **Duplicate rows and the split.** The cleaned data contains 42 rows forming 20
   groups of fully-identical feature+label records. The **cleaning report counts
   and documents them without dropping** (OpenML omits `customerID`). However, an
   audit found that under the pinned seed **7 of those groups straddled the
   train/test split** — i.e. 7 test rows had an identical twin in the training
   set, causing minor memorized-row leakage into the test evaluation. This is
   fixed by **deduplicating exact rows before the split**
   (`deduplicate_for_split`, dropping 22 rows; ~0.3% of data), so no identical
   row can appear on both sides. Enforced by
   `tests/test_split_duplicates.py::test_no_duplicate_group_straddles_split`.

---

## License / data

Data: IBM Telco Customer Churn via [OpenML dataset 42178](https://www.openml.org/d/42178)
(public). See `data/raw/DATA_MANIFEST.json` for the exact version and checksum.
