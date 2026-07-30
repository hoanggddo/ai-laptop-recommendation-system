"""Data loading, cleaning, and scoring logic for the AI Laptop Recommender.

Kept separate from app.py so the data/scoring logic can be tested or reused
without spinning up Streamlit, and separate from model.py so the ML model
doesn't need to know anything about pandas/CSV parsing.
"""
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Static conversion rate for demo purposes — not a live exchange rate.
INR_TO_USD = 0.012


def load_data(csv_path: str = "laptops.csv"):
    """Load laptops.csv, clean it, and encode categorical columns.

    Returns:
        (data, encoders): the cleaned DataFrame, and a dict of fitted
        LabelEncoders for any categorical columns that were encoded.
    """
    data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    data.dropna(inplace=True)
    data.columns = data.columns.str.strip()

    data["ram"] = data["ram"].apply(_extract_int)
    data["storage"] = data["storage"].apply(_extract_int)
    data["display(in inch)"] = pd.to_numeric(data["display(in inch)"], errors="coerce")

    encoders = {}
    for col in ("processor", "os"):
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    data["price_usd"] = data["price(in Rs.)"] * INR_TO_USD
    return data, encoders


def _extract_int(value) -> int:
    """Pull the first integer out of a string like '8 GB' -> 8."""
    match = re.findall(r"\d+", str(value))
    return int(match[0]) if match else 0


def compute_score(row, desired_specs: dict, budget_usd: float) -> float:
    """Weighted match score: 50% spec fit, 30% rating, 20% budget fit.

    Clamped to [0, 1] so it's always safe to pass to st.progress().
    """
    spec_cols = {"ram": "ram", "storage": "storage", "display": "display(in inch)"}
    spec_score = sum(
        min(row[spec_cols[k]] / max(desired_specs[k], 1), 1) for k in desired_specs
    ) / len(desired_specs)

    rating_score = row["rating"] / 5
    budget_score = max(
        0.0, 1 - abs(row["price_usd"] - budget_usd) / max(budget_usd, 1)
    )

    return 0.5 * spec_score + 0.3 * rating_score + 0.2 * budget_score


def get_bar_color(value: float, desired: float) -> str:
    """Traffic-light color for a spec bar relative to the desired value."""
    ratio = value / max(desired, 1)
    if ratio >= 1.0:
        return "green"
    elif ratio >= 0.8:
        return "yellow"
    return "red"
