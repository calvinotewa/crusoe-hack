#!/usr/bin/env python3
"""Generate a synthetic customer-churn dataset for the Shared Brain harness.

The data-generating process (DGP) is a logistic model — churn probability rises with
support calls and monthly charges, falls with tenure, satisfaction and commitment
(contract length). The harness benchmark (RMSE against the 0/1 churn label) is then a
Brier score, a proper scoring rule for probabilistic classifiers, so "lower RMSE =
better churn model" is meaningful.

    uv run python examples/make_churn_data.py [rows] [seed]
"""

import csv
import math
import random
import sys
from pathlib import Path


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# Ground-truth DGP: logit(churn). Agents never see these; the curator's job is to
# rediscover this structure across generations.
COEFFS = {
    "intercept": -3.0,
    "support_calls": 0.55,  # more calls -> angry customer -> churn risk up
    "monthly_charges": 0.018,  # price sensitivity
    "tenure_months": -0.045,  # loyalty
    "satisfaction": -0.75,  # 1..5 scale, strong negative driver
    "contract_months": -0.06,  # commitment
    "support_x_charges": 0.006,  # interaction: expensive AND calling support
}


def churn_logit(row: dict) -> float:
    return (
        COEFFS["intercept"]
        + COEFFS["support_calls"] * row["support_calls"]
        + COEFFS["monthly_charges"] * row["monthly_charges"]
        + COEFFS["tenure_months"] * row["tenure_months"]
        + COEFFS["satisfaction"] * row["satisfaction"]
        + COEFFS["contract_months"] * row["contract_months"]
        + COEFFS["support_x_charges"] * row["support_calls"] * row["monthly_charges"]
    )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        row = {
            "tenure_months": rng.randint(1, 72),
            "monthly_charges": round(rng.uniform(15, 120), 2),
            "support_calls": rng.randint(0, 12),
            "satisfaction": rng.randint(1, 5),
            "contract_months": rng.choice([1, 12, 24]),
        }
        p = sigmoid(churn_logit(row))
        row["y"] = 1 if rng.random() < p else 0  # observed churn label (0/1)
        rows.append(row)

    out = Path(__file__).parent / "churn_synthetic.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Reference points for judging harness quality:
    ys = [r["y"] for r in rows]
    ps = [sigmoid(churn_logit(r)) for r in rows]
    mean = sum(ys) / n
    const_brier = sum((mean - y) ** 2 for y in ys) / n  # naive constant predictor
    bayes_brier = sum(p * (1 - p) for p in ps) / n  # DGP oracle (irreducible)
    print(f"wrote {out} ({n} rows, seed {seed})")
    print(f"base churn rate: {mean:.3f}")
    naive_score = 1 / (1 + math.sqrt(const_brier))
    oracle_score = 1 / (1 + math.sqrt(bayes_brier))
    print(f"constant-predictor Brier (RMSE^2): {const_brier:.4f} -> score {naive_score:.4f}")
    print(f"oracle (DGP) Brier floor:         {bayes_brier:.4f} -> score {oracle_score:.4f}")


if __name__ == "__main__":
    main()
