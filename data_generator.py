import pandas as pd
import numpy as np

def generate_credit_data(n=6000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        "age": np.random.randint(21, 65, n),
        "annual_income": np.random.randint(20000, 150000, n),
        "credit_score": np.random.randint(300, 850, n),
        "loan_amount": np.random.randint(1000, 50000, n),
        "loan_term_months": np.random.choice([12, 24, 36, 48, 60], n),
        "employment_years": np.random.randint(0, 20, n),
        "debt_to_income_ratio": np.round(np.random.uniform(0.1, 0.8, n), 2),
        "num_credit_cards": np.random.randint(0, 10, n),
        "has_previous_default": np.random.choice([0, 1], n, p=[0.85, 0.15])
    })

    # Risk Logic
    risk_score = (
        -0.005 * df["credit_score"] +
        0.00002 * df["loan_amount"] +
        2.5 * df["debt_to_income_ratio"] -
        0.00001 * df["annual_income"] +
        1.5 * df["has_previous_default"] -
        0.05 * df["employment_years"]
    )

    prob = 1 / (1 + np.exp(-risk_score))
    df["loan_default"] = np.random.binomial(1, prob)

    return df

if __name__ == "__main__":
    df = generate_credit_data()
    df.to_csv("data/credit_data.csv", index=False)