# Exploratory Data Analysis

Overall churn rate: **26.5%** (1869 of 7043).
Each section answers a specific business question.

## 1. Does contract type drive churn?
![contract](figures/churn_by_contract.png)
- Month-to-month churns far more than fixed-term. Highest group: **Month-to-month** at 42.7%.

## 2. Is churn front-loaded in tenure?
![tenure](figures/churn_by_tenure.png)
- Churn is concentrated in the first year (**0-12** bucket: 47.4%), then falls with tenure.

## 3. Are customers price-sensitive?
![monthly](figures/churn_by_monthly.png)
- Churn rises with monthly charges, consistent with price sensitivity.

## 4. Does payment method matter?
![payment](figures/churn_by_payment.png)
- **Electronic check** has the highest churn (45.3%) — the electronic-check pattern.

## 5. Fiber vs DSL
![internet](figures/churn_by_internet.png)
- Highest-churn internet type: **Fiber optic** (41.9%).

## 6. Do add-on services retain?
![tech](figures/churn_by_techsupport.png)
- Customers without tech support churn more, suggesting add-ons aid retention.

## 7. Collinearity trio
- `TotalCharges` ~ `tenure x MonthlyCharges` (see data quality report);
  interpret these three as one correlated bundle.