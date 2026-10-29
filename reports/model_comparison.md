# Model Comparison

## Methodology

Model selection, calibration, and the profit threshold are all decided
using **stratified 5-fold CV on the training set only** (seeded). Out-of-
fold probabilities are produced from the **calibrated** estimator, so the
threshold chosen on OOF transfers to the calibrated test predictions on
the same scale. The held-out 20% test set is scored **once** to produce
the headline numbers. CV scores drive selection; **test-set numbers are
the headline.**

## Cross-validation scores — pooled out-of-fold (training set)

| model         |   roc_auc |   pr_auc |   recall@p0.3 |   recall@p0.5 |   recall@p0.7 |
|:--------------|----------:|---------:|--------------:|--------------:|--------------:|
| logreg        |    0.8432 |   0.6505 |        0.9906 |        0.8168 |        0.4357 |
| random_forest |    0.8434 |   0.6497 |        0.9912 |        0.8269 |        0.437  |
| xgboost       |    0.8346 |   0.6388 |        0.9919 |        0.8054 |        0.3879 |

## Cross-validation scores — per-fold mean ± std (training set)

| model         |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   recall@p0.3_mean |   recall@p0.3_std |   recall@p0.5_mean |   recall@p0.5_std |   recall@p0.7_mean |   recall@p0.7_std |
|:--------------|---------------:|--------------:|--------------:|-------------:|-------------------:|------------------:|-------------------:|------------------:|-------------------:|------------------:|
| logreg        |         0.8446 |        0.0084 |        0.6546 |       0.0299 |             0.9872 |            0.0055 |             0.8263 |            0.013  |             0.4276 |            0.0991 |
| random_forest |         0.8448 |        0.0103 |        0.653  |       0.0273 |             0.9906 |            0.0044 |             0.829  |            0.0159 |             0.4162 |            0.1298 |
| xgboost       |         0.836  |        0.0093 |        0.643  |       0.023  |             0.9919 |            0.0051 |             0.8074 |            0.0182 |             0.3906 |            0.1001 |

**Selected model (max CV PR-AUC): `logreg`**

## Held-out test-set scores (headline)

- roc_auc: **0.8398**
- pr_auc: **0.6326**
- recall@p0.3: **0.9946**
- recall@p0.5: **0.7715**
- recall@p0.7: **0.4704**

## Calibration (Brier score, lower is better)

- uncalibrated: **0.1668**
- calibrated: **0.1384**

See `figures/calibration.png` for reliability curves (before/after
calibration).