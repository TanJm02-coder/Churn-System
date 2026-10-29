# Churn Drivers — Plain-Language Summary

> **Note:** SHAP here explains the best *tree* model (`random_forest`), which is **not** the deployed scoring model (`logreg`). TreeExplainer only supports tree ensembles, so when a non-tree model wins selection, SHAP is computed on the best tree model as a directional guide to drivers — treat it as indicative of churn structure, not an exact explanation of the scorer.

Ranked by mean absolute SHAP value on a **training-set** sample. SHAP
measures how much each feature pushes an individual prediction toward or
away from churn.

## Top drivers

- **cat__Contract_Month-to-month** (mean |SHAP| = 0.0763)
- **num__tenure** (mean |SHAP| = 0.0544)
- **cat__InternetService_Fiber optic** (mean |SHAP| = 0.0452)
- **num__TotalCharges** (mean |SHAP| = 0.0362)
- **cat__Contract_Two year** (mean |SHAP| = 0.0342)
- **cat__OnlineSecurity_No** (mean |SHAP| = 0.0334)
- **cat__TechSupport_No** (mean |SHAP| = 0.0281)
- **cat__PaymentMethod_Electronic check** (mean |SHAP| = 0.0252)

## How to read this

- Short-tenure, month-to-month customers and those on electronic-check
  payment typically show the strongest churn signal in this dataset.
- Longer contracts and value-added services (tech support, online
  security) are associated with retention.

## Caveat: collinearity

`TotalCharges` is approximately `tenure x MonthlyCharges`. Because these
three move together, SHAP importance can be split or shifted among them;
do not read any single one's importance in isolation. Treat tenure,
monthly charges, and total charges as one correlated bundle.