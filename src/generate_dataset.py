from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def main():
    rng = np.random.default_rng(42)
    rows = 5000

    tenure_months = rng.integers(1, 73, rows)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.28, 0.17])
    internet_service = rng.choice(["fiber", "dsl", "none"], rows, p=[0.52, 0.36, 0.12])
    autopay = rng.choice([0, 1], rows, p=[0.42, 0.58])
    paperless_billing = rng.choice([0, 1], rows, p=[0.35, 0.65])
    streaming_services = rng.integers(0, 5, rows)
    support_tickets = rng.poisson(1.3, rows).clip(0, 8)
    late_payments = rng.poisson(0.45, rows).clip(0, 6)
    age = rng.integers(18, 78, rows)
    household_size = rng.integers(1, 6, rows)
    region = rng.choice(["Northeast", "South", "Midwest", "West"], rows)

    base_charge = rng.normal(62, 15, rows)
    monthly_charges = (
        base_charge
        + np.where(internet_service == "fiber", 28, 0)
        + np.where(internet_service == "dsl", 12, 0)
        + streaming_services * 7
        + rng.normal(0, 5, rows)
    ).clip(20, 165).round(2)

    satisfaction_score = (
        8.2
        - support_tickets * 0.55
        - late_payments * 0.25
        - np.where(contract_type == "month-to-month", 0.35, 0)
        + np.where(autopay == 1, 0.25, 0)
        + rng.normal(0, 0.9, rows)
    ).clip(1, 10).round(1)

    churn_logit = (
        -2.2
        + np.where(contract_type == "month-to-month", 1.2, 0)
        + np.where(contract_type == "two-year", -0.9, 0)
        + np.where(internet_service == "fiber", 0.35, 0)
        + support_tickets * 0.28
        + late_payments * 0.33
        + (monthly_charges - 70) * 0.012
        - tenure_months * 0.018
        - autopay * 0.55
        - (satisfaction_score - 6.5) * 0.45
        + rng.normal(0, 0.35, rows)
    )

    churn_probability = sigmoid(churn_logit)
    churned = rng.binomial(1, churn_probability)

    data = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, rows + 1)],
            "region": region,
            "age": age,
            "household_size": household_size,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "autopay": autopay,
            "paperless_billing": paperless_billing,
            "streaming_services": streaming_services,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "satisfaction_score": satisfaction_score,
            "churned": churned,
        }
    )

    data.to_csv(DATA_DIR / "customer_churn.csv", index=False)
    print(f"Generated {len(data):,} rows at {DATA_DIR / 'customer_churn.csv'}")


if __name__ == "__main__":
    main()
