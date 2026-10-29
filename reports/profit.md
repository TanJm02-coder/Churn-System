# Profit Layer (Simulated Expected Value)

> **All figures here are a simulated expected value under assumed
> economics, not measured profit.**

## Assumptions (defaults; adjustable in the app)
- Offer cost: $50
- Offer success rate: 30%
- Gross margin: 30%
- Expected lifetime cap: 24 months
- Annual discount rate: 10%

## Threshold selection
- Threshold chosen on **out-of-fold training predictions**, then applied
  **once on the held-out test set**.
- Chosen threshold: **0.365**
- Customers targeted on test set: **393** of 1405
- **Headline simulated EV on test set: $17,601**
- **Bootstrap 95% CI: [$15,643, $19,714]** (1000 resamples of test customers; threshold held fixed)

## Model-targeting uplift vs naive policies

- Target **no one**: $0
- Target **everyone**: $-16,471
- Model-chosen threshold: **$17,601** (**+$34,072** vs targeting everyone)

The model-driven policy beats both naive extremes: it avoids wasting
offer cost on low-risk customers (unlike target-everyone) while still
capturing most retainable value (unlike target-no-one).

## Threshold robustness (neighborhood of the optimum)

|   threshold |   profit |   pct_of_best |
|------------:|---------:|--------------:|
|       0.265 |  16648.3 |          94.6 |
|       0.29  |  16972.7 |          96.4 |
|       0.315 |  17236.5 |          97.9 |
|       0.34  |  17290.6 |          98.2 |
|       0.365 |  17601.1 |         100   |
|       0.39  |  17761.7 |         100.9 |
|       0.415 |  17333.9 |          98.5 |
|       0.44  |  17188.7 |          97.7 |
|       0.465 |  16362.4 |          93   |

A broad plateau near 100% of best profit indicates the optimum is robust,
not a fragile knife-edge; small threshold mis-estimates cost little.

## Sensitivity (offer cost x success rate)

|   offer_cost |   success_rate |   best_threshold |   best_profit |   n_targeted | targeting_pays_off   |
|-------------:|---------------:|-----------------:|--------------:|-------------:|:---------------------|
|           25 |           0.15 |             0.39 |       8880.86 |          380 | True                 |
|           25 |           0.3  |             0.19 |      31030.2  |          657 | True                 |
|           25 |           0.45 |             0.15 |      55437.5  |          794 | True                 |
|           25 |           0.6  |             0.09 |      80820    |          942 | True                 |
|           50 |           0.15 |             0.68 |       1867.01 |          121 | True                 |
|           50 |           0.3  |             0.39 |      17761.7  |          380 | True                 |
|           50 |           0.45 |             0.24 |      38844.4  |          588 | True                 |
|           50 |           0.6  |             0.19 |      62060.4  |          657 | True                 |
|           75 |           0.15 |             0.85 |        193.76 |           27 | True                 |
|           75 |           0.3  |             0.54 |       9271.55 |          248 | True                 |
|           75 |           0.45 |             0.39 |      26642.6  |          380 | True                 |
|           75 |           0.6  |             0.28 |      47037.1  |          528 | True                 |
|          100 |           0.15 |             0.97 |        -16.81 |            2 | False                |
|          100 |           0.3  |             0.68 |       3734.02 |          121 | True                 |
|          100 |           0.45 |             0.44 |      17649.3  |          316 | True                 |
|          100 |           0.6  |             0.39 |      35523.4  |          380 | True                 |
|          150 |           0.15 |             0.97 |       -116.81 |            2 | False                |
|          150 |           0.3  |             0.85 |        387.51 |           27 | True                 |
|          150 |           0.45 |             0.68 |       5601.03 |          121 | True                 |
|          150 |           0.6  |             0.54 |      18543.1  |          248 | True                 |

Rows with `targeting_pays_off = False` mark the regime where the program
is net-negative under those assumptions.