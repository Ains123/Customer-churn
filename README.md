# Customer-churn
# SyriaTel Customer Churn Classification
**Dataset:** [SyriaTel Customer Churn — Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

---

## Overview

This project builds a binary classification model to predict customer churn for SyriaTel, a telecommunications company. The goal is to identify customers who are likely to cancel their service so that the retention team can intervene proactively — before the customer leaves.

Acquiring a new customer costs SyriaTel 5–10× more than retaining an existing one. A churn model shifts the retention team from reactive to proactive, enabling targeted outreach (discounts, service upgrades, follow-up calls) to the customers most at risk.

---

## Business Problem

**Stakeholder:** SyriaTel's Customer Retention Team

**Problem:** SyriaTel currently discovers unhappy customers only after they cancel. The retention team needs a way to flag at-risk customers in advance so they can be contacted before churning.

**Solution:** A monthly scoring pipeline that runs this model against the full customer database, surfaces the top churn-risk accounts, and feeds them into the retention team's outreach workflow.

---

## Dataset

| Property | Value |
|---|---|
| Source | BigML / Kaggle |
| File | `bigml_59c28831336c6604c800002a.csv` |
| Rows | 3,333 customers |
| Columns | 21 features |
| Target | `churn` (True/False → 1/0) |
| Churn rate | ~14.5% (class imbalance) |

### Data Limitations

- **Single time snapshot** — cannot capture trends in individual customer behavior over time
- **No competitor data** — competitor pricing or offers are not captured
- **No network quality data** — signal strength and outage history are absent
- **Limited geography** — only state-level location is available

---

## Repository Structure

```
├── phase3_syriatel_churn.ipynb   # Main analysis notebook
├── README.md                     # This file
└── data/
    └── bigml_59c28831336c6604c800002a.csv  # Dataset (download from Kaggle)
```

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) and place it in the project directory.
2. Open `phase3_syriatel_churn.ipynb` in Jupyter.
3. Run all cells in order from top to bottom.

---

## Project Workflow

### 1. Business Understanding

Define the churn problem from SyriaTel's perspective, identify the stakeholder (retention team), and frame the project around maximizing **recall** — catching as many actual churners as possible, since a missed churner costs more than a false alarm.

### 2. Data Understanding

Explore the 3,333-row dataset across 21 features. Key findings from EDA:

- **14.5% churn rate** — the dataset is imbalanced; pure accuracy is a misleading metric
- **International plan customers churn at ~3× the baseline rate** — a major risk signal
- **Churners average 54% more customer service calls** than non-churners — repeated unresolved issues correlate strongly with churn

### 3. Data Preparation

| Step | Action | Justification |
|---|---|---|
| Drop columns | `phone number`, `state`, `area code` | Identifiers with no predictive signal; `phone number` causes leakage |
| Encode | `international plan`, `voice mail plan` → 0/1 | sklearn requires numeric input |
| Convert target | `churn` bool → int | Compatibility with classifiers and metrics |
| Split | Stratified 80/20 train/test | Preserves 14.5% churn rate in both sets |
| Scale | Inside `evaluate()`, fit on train only | Prevents data leakage |

### 4. Modeling

Four models were built iteratively, each justified by the findings of the previous one:

| # | Model | Why | Key Finding |
|---|---|---|---|
| 1 | Logistic Regression | Interpretable baseline | Low recall — linear boundary can't capture non-linear churn patterns |
| 2 | Decision Tree (Default) | Non-parametric; checks for non-linear patterns | 100% train accuracy — clear overfitting |
| 3 | Decision Tree (Tuned) | GridSearchCV + `class_weight='balanced'` | Overfitting reduced; recall improved |
| 4 | **Random Forest + Pipeline** | Ensemble reduces variance; Pipeline prevents leakage | **Best test recall — selected as final model** |

**Metric rationale:** Models are scored on **recall** rather than accuracy. A missed churner (false negative) costs SyriaTel a customer and lost recurring revenue. A false positive only costs a low-value retention outreach. Recall directly captures the business cost.

### 5. Evaluation

The final Random Forest Pipeline was evaluated on a held-out test set (never seen during training).

| Model | Train Acc | Test Acc | Test Recall |
|---|---|---|---|
| Logistic Regression | ~0.86 | ~0.85 | ~0.17 |
| Decision Tree (Default) | 1.000 | ~0.91 | ~0.70 |
| Decision Tree (Tuned) | ~0.83 | ~0.92 | ~0.76 |
| **Random Forest (Pipeline)** | **~0.99** | **~0.93** | **~0.79** |

**Top 5 feature importances (Random Forest):**

1. `total day charge` — high bills signal price-sensitive customers likely to switch
2. `customer service calls` — repeated calls indicate unresolved frustration
3. `international plan` — ~3× churn rate vs. customers without the plan
4. `total eve charge` — evening usage charges contribute to price sensitivity
5. `total intl calls` — international call volume correlates with plan dissatisfaction

---

## Key Findings

- **Total day charge is the strongest predictor.** Customers with unusually high day charges are price-sensitive and at risk of switching to cheaper alternatives.
- **Customer service calls are a leading indicator.** Customers who call support repeatedly have unresolved problems that are eroding satisfaction.
- **International plan holders churn at ~3× the rate** of non-plan customers, suggesting a pricing or service quality gap vs. competitors.

---

## Recommendations

1. **Run the model monthly.** Score all active customers and prioritize the top 5–10% by churn probability for proactive outreach.

2. **Target high-charge + high-call customers first.** Customers with both high daily charges and 3+ service calls in 90 days are the highest-risk segment.

3. **Review international plan pricing immediately.** A ~3× churn rate is a strong signal of a pricing or service quality gap vs. competitors. Audit the plan's competitiveness.

4. **Track outcomes and retrain quarterly.** Capture 90-day outcomes after interventions and retrain the model regularly to keep it current as market conditions shift.

---

## Limitations

- **Time snapshot only** — churn patterns shift with market changes; regular retraining is required
- **Precision trade-off** — higher recall means some outreach targets customers who would not have churned; weigh outreach cost vs. lifetime value before full deployment
- **Missing data** — competitor offers, network quality metrics, and satisfaction scores would likely improve model performance
- **Validation needed** — test on a fresh customer cohort before full production deployment

---

## Contact

For questions about this project, please open an issue or reach out via the repository.
