# Customer Segments (k-means)

k selected by silhouette score. Per-segment model performance includes
sample size and bootstrap 95% CIs; `low_confidence = True` flags segments
with few positives where PR-AUC is unstable.

## Segment profiles

|   segment |   n |   churn_rate |   avg_monthly |   avg_tenure | top_contract   | top_payment             |
|----------:|----:|-------------:|--------------:|-------------:|:---------------|:------------------------|
|         0 | 643 |       0.4417 |         67.37 |         15.4 | Month-to-month | Electronic check        |
|         1 | 298 |       0.0906 |         21.25 |         32.6 | Two year       | Mailed check            |
|         2 | 464 |       0.1315 |         88.85 |         56.5 | Two year       | Credit card (automatic) |

## Model performance by segment (test set)

|   segment |   n |   n_positives |   pr_auc |   pr_auc_ci_lo |   pr_auc_ci_hi | low_confidence   |
|----------:|----:|--------------:|---------:|---------------:|---------------:|:-----------------|
|         0 | 643 |           284 |   0.683  |         0.6284 |         0.7446 | False            |
|         1 | 298 |            27 |   0.4101 |         0.2612 |         0.5944 | True             |
|         2 | 464 |            61 |   0.4187 |         0.3004 |         0.5595 | False            |