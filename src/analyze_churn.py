import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
MPL_DIR = ROOT / ".mplconfig"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = ROOT / "data" / "customer_churn.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "age",
    "household_size",
    "tenure_months",
    "monthly_charges",
    "autopay",
    "paperless_billing",
    "streaming_services",
    "support_tickets",
    "late_payments",
    "satisfaction_score",
]
CATEGORICAL_FEATURES = ["region", "contract_type", "internet_service"]
TARGET = "churned"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pipeline": model,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out()


def get_feature_importance(best_result):
    pipeline = best_result["pipeline"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["feature"] = (
        importance["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
    )
    return importance.sort_values("importance", ascending=False).head(12)


def save_bar_chart(frame, x_col, y_col, title, path, color="#087f8c"):
    plt.figure(figsize=(9, 5))
    plt.bar(frame[x_col], frame[y_col], color=color)
    plt.title(title)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_dashboard(metrics, segment_tables, top_features):
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Customer Churn Analytics</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #16202a; background: #f6faf9; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin-bottom: 6px; font-size: 34px; }}
    .sub {{ color: #5d6c7b; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 24px 0; }}
    .card {{ background: white; border: 1px solid #dce5e8; border-radius: 8px; padding: 18px; box-shadow: 0 12px 32px rgba(22,32,42,.07); }}
    .metric {{ font-size: 30px; font-weight: 800; color: #087f8c; margin: 0; }}
    .label {{ color: #647282; margin: 6px 0 0; font-weight: 700; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #dce5e8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5ecef; padding: 9px; text-align: left; }}
    th {{ color: #087f8c; }}
    @media (max-width: 900px) {{ .grid, .two {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Customer Churn Analytics & Retention Modeling</h1>
  <p class="sub">End-to-end data science project: churn analysis, segmentation, predictive modeling, and retention recommendations.</p>

  <section class="grid">
    <div class="card"><p class="metric">{metrics['customers']:,}</p><p class="label">Customers</p></div>
    <div class="card"><p class="metric">{metrics['churn_rate']:.1%}</p><p class="label">Churn Rate</p></div>
    <div class="card"><p class="metric">${metrics['monthly_revenue_at_risk']:,.0f}</p><p class="label">Monthly Revenue At Risk</p></div>
    <div class="card"><p class="metric">{metrics['best_auc']:.3f}</p><p class="label">Best ROC AUC</p></div>
  </section>

  <section class="two">
    <div class="card"><h2>Churn by Contract</h2><img src="figures/churn_by_contract.png" alt="Churn by contract chart"></div>
    <div class="card"><h2>Top Model Drivers</h2><img src="figures/top_features.png" alt="Feature importance chart"></div>
    <div class="card"><h2>Revenue at Risk</h2><img src="figures/revenue_at_risk.png" alt="Revenue risk chart"></div>
    <div class="card"><h2>Confusion Matrix</h2><img src="figures/confusion_matrix.png" alt="Confusion matrix"></div>
  </section>

  <section class="two" style="margin-top: 16px;">
    <div class="card">
      <h2>Highest-Risk Segments</h2>
      {segment_tables['risk']}
    </div>
    <div class="card">
      <h2>Top Predictive Features</h2>
      {top_features.to_html(index=False, classes='table')}
    </div>
  </section>
</main>
</body>
</html>
"""
    (REPORT_DIR / "dashboard.html").write_text(html, encoding="utf-8")


def main():
    data = pd.read_csv(DATA_PATH)
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    models = [
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=9,
                            min_samples_leaf=8,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
    ]

    results = [evaluate_model(name, model, x_train, x_test, y_train, y_test) for name, model in models]
    metrics_table = pd.DataFrame([{k: v for k, v in result.items() if k not in ["pipeline", "predictions", "probabilities"]} for result in results])
    best_result = max(results, key=lambda item: item["roc_auc"])
    best_pipeline = best_result["pipeline"]

    data["predicted_churn_probability"] = best_pipeline.predict_proba(x)[:, 1]
    data["revenue_at_risk"] = data["monthly_charges"] * data["predicted_churn_probability"]

    top_features = get_feature_importance(best_result)
    top_features.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    metrics_table.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    data.to_csv(REPORT_DIR / "scored_customers.csv", index=False)

    contract_churn = data.groupby("contract_type", as_index=False)["churned"].mean()
    save_bar_chart(contract_churn, "contract_type", "churned", "Churn Rate by Contract Type", FIGURE_DIR / "churn_by_contract.png")

    save_bar_chart(top_features.sort_values("importance"), "feature", "importance", "Top Predictive Features", FIGURE_DIR / "top_features.png", color="#d95d39")

    revenue_risk = (
        data.groupby("contract_type", as_index=False)["revenue_at_risk"]
        .sum()
        .sort_values("revenue_at_risk", ascending=False)
    )
    save_bar_chart(revenue_risk, "contract_type", "revenue_at_risk", "Predicted Monthly Revenue at Risk", FIGURE_DIR / "revenue_at_risk.png", color="#d7a21f")

    ConfusionMatrixDisplay.from_predictions(y_test, best_result["predictions"], cmap="Blues")
    plt.title(f"{best_result['model']} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    risk_segments = (
        data.assign(risk_band=pd.cut(data["predicted_churn_probability"], bins=[0, 0.35, 0.65, 1], labels=["Low", "Medium", "High"]))
        .groupby(["contract_type", "risk_band"], observed=True)
        .agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"), revenue_at_risk=("revenue_at_risk", "sum"))
        .reset_index()
        .sort_values(["churn_rate", "revenue_at_risk"], ascending=False)
        .head(8)
    )

    metrics = {
        "customers": len(data),
        "churn_rate": data["churned"].mean(),
        "monthly_revenue_at_risk": data["revenue_at_risk"].sum(),
        "best_auc": best_result["roc_auc"],
    }

    build_dashboard(metrics, {"risk": risk_segments.to_html(index=False)}, top_features)

    summary = f"""# Model Summary

## Dataset

- Customers analyzed: {len(data):,}
- Overall churn rate: {data['churned'].mean():.1%}
- Predicted monthly revenue at risk: ${data['revenue_at_risk'].sum():,.0f}

## Best Model

- Model: {best_result['model']}
- ROC AUC: {best_result['roc_auc']:.3f}
- F1 score: {best_result['f1']:.3f}
- Precision: {best_result['precision']:.3f}
- Recall: {best_result['recall']:.3f}

## Business Recommendations

- Prioritize month-to-month customers with high support ticket volume and low satisfaction scores.
- Promote autopay and annual contract incentives to reduce voluntary churn risk.
- Build a retention queue using predicted churn probability and monthly revenue at risk.
- Track retention campaign impact with churn rate, saved revenue, and customer satisfaction lift.
"""
    (REPORT_DIR / "model_summary.md").write_text(summary, encoding="utf-8")
    print(f"Best model: {best_result['model']} ROC AUC={best_result['roc_auc']:.3f}")
    print(f"Dashboard written to {REPORT_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
