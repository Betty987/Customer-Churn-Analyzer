# Customer Churn Analytics & Retention Modeling

This project analyzes customer churn for a subscription business and builds a predictive model to identify high-risk customers. It combines exploratory data analysis, segmentation, machine learning, feature importance, and business recommendations.

The goal is to show the full data science workflow: define the business problem, prepare data, model churn risk, interpret drivers, and translate results into retention actions.

## Project Highlights

- Generated and analyzed a 5,000-customer subscription dataset
- Built churn models with logistic regression and random forest classifiers
- Evaluated models with ROC AUC, precision, recall, F1 score, and confusion matrix
- Scored customers by predicted churn probability and monthly revenue at risk
- Identified high-risk customer segments by contract type and churn band
- Created a static HTML dashboard with charts and model results

## Business Question

Which customers are most likely to churn, what factors drive churn risk, and where should the retention team focus first?

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib
- HTML report generation

## Repository Structure

```text
customer-churn-analytics/
  data/
    customer_churn.csv
  reports/
    dashboard.html
    model_metrics.csv
    feature_importance.csv
    model_summary.md
    scored_customers.csv
    figures/
  src/
    generate_dataset.py
    analyze_churn.py
  requirements-dev.txt
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Generate the dataset:

```bash
python src/generate_dataset.py
```

Run the analysis:

```bash
python src/analyze_churn.py
```

Open the dashboard:

```text
reports/dashboard.html
```

## Outputs

- `reports/dashboard.html`: visual dashboard with KPIs, charts, and segment tables
- `reports/model_summary.md`: business summary and model performance
- `reports/model_metrics.csv`: model comparison table
- `reports/feature_importance.csv`: top churn drivers
- `reports/scored_customers.csv`: customer-level predicted churn probabilities

