# Fairness Audit (Report & Discuss Only)

Error rates computed at the profit threshold **0.365** on the
test set. **No mitigation is applied** — this is a reporting and
discussion layer.

Each rate includes group sample size `n` and bootstrap 95% CIs for recall
and selection rate; narrow groups are low-confidence.

| attribute     | group   |    n |   recall |    fpr |    fnr |   selection_rate |   recall_ci_lo |   recall_ci_hi |   selection_ci_lo |   selection_ci_hi |
|:--------------|:--------|-----:|---------:|-------:|-------:|-----------------:|---------------:|---------------:|------------------:|------------------:|
| gender        | Female  |  671 |   0.6368 | 0.1601 | 0.3632 |           0.2951 |         0.5684 |         0.7054 |            0.2623 |            0.3294 |
| gender        | Male    |  734 |   0.5879 | 0.1594 | 0.4121 |           0.2657 |         0.511  |         0.6538 |            0.2343 |            0.297  |
| SeniorCitizen | 0       | 1176 |   0.5583 | 0.1277 | 0.4417 |           0.2313 |         0.5017 |         0.6148 |            0.2092 |            0.2577 |
| SeniorCitizen | 1       |  229 |   0.7865 | 0.3643 | 0.2135 |           0.5284 |         0.6963 |         0.8652 |            0.4672 |            0.5895 |
| Partner       | No      |  736 |   0.6708 | 0.2312 | 0.3292 |           0.3764 |         0.6091 |         0.7285 |            0.341  |            0.4104 |
| Partner       | Yes     |  669 |   0.5039 | 0.0944 | 0.4961 |           0.1734 |         0.4186 |         0.5969 |            0.145  |            0.2003 |
| Dependents    | No      |  974 |   0.6555 | 0.2119 | 0.3445 |           0.348  |         0.5987 |         0.709  |            0.3183 |            0.3799 |
| Dependents    | Yes     |  431 |   0.4384 | 0.0615 | 0.5616 |           0.1253 |         0.3284 |         0.5479 |            0.0928 |            0.1601 |

## Discussion

- **`gender` is excluded from the model's features by design**, yet it is
  audited here. Excluding it does **not** remove proxy effects: the model
  can still encode group information through correlated features, and
  `SeniorCitizen`, `Partner`, and `Dependents` remain model features and
  may act as proxies.
- Compare FPR/FNR/recall across groups. A higher FNR for a group means
  that group's at-risk customers are under-targeted for retention offers;
  a higher selection rate means more offers (and cost) directed at it.
- Mitigation (e.g. group-specific thresholds) is deliberately omitted: it
  introduces its own fairness tradeoffs and is out of scope by agreement.

## Notes on method

- `SeniorCitizen` is the one sensitive attribute stored as 0/1 and is
  treated as a numeric feature by the model; grouping here uses its raw
  0/1 values, so the audit is unaffected.
- `fnr` is reported alongside `recall` for readability even though
  `fnr = 1 - recall`; both are derived from the same confusion cells.
- `n` is the group size; recall and selection-rate CIs are percentile
  bootstraps and will be wide for small groups (flagged as low-confidence
  in the segmentation report).