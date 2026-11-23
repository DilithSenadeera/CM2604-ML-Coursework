"""Read, validate and split the original Telco Customer Churn dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SEED = 42
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
AUDIT_COLUMNS = ["gender", "SeniorCitizen"]
CATEGORICAL_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path=DATA_FILE):
    """Return prepared customer records and the count of source charge blanks."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = set(NUMERIC_COLUMNS + AUDIT_COLUMNS + CATEGORICAL_COLUMNS + ["customerID", "Churn"])
    if set(raw.columns) != expected:
        raise ValueError("CSV columns do not match the expected Kaggle dataset schema.")

    customers = raw.copy()
    for column in customers.columns:
        customers[column] = customers[column].str.strip()

    if (customers["customerID"] == "").any() or customers["customerID"].duplicated().any():
        raise ValueError("Every row must have a unique, nonempty customer ID.")
    if set(customers["Churn"]) != {"Yes", "No"}:
        raise ValueError("Expected both churn labels, Yes and No.")
    if not customers["SeniorCitizen"].isin(["0", "1"]).all():
        raise ValueError("Unexpected senior-citizen flag.")
    other_columns = customers.drop(columns="TotalCharges")
    if (other_columns == "").any().any():
        raise ValueError("Unexpected blank outside TotalCharges.")

    customers["tenure"] = pd.to_numeric(customers["tenure"])
    blank_total = customers["TotalCharges"] == ""
    if (blank_total & (customers["tenure"] != 0)).any():
        raise ValueError("Review blank total charges on accounts with positive tenure.")
    # For zero-tenure accounts only, assume no cumulative charges when the total is blank.
    customers.loc[blank_total, "TotalCharges"] = "0"
    customers["TotalCharges"] = pd.to_numeric(customers["TotalCharges"])
    customers["MonthlyCharges"] = pd.to_numeric(customers["MonthlyCharges"])
    numeric = customers[NUMERIC_COLUMNS].to_numpy()
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Tenure and charges must be finite and nonnegative.")
    if (customers["tenure"] % 1 != 0).any():
        raise ValueError("Expected tenure in whole months.")

    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})
    return customers, int(blank_total.sum())


def split_data(customers):
    """Return the same stratified development and test rows for every stage."""
    return train_test_split(
        customers, test_size=0.20, random_state=SEED, stratify=customers["Churn"]
    )
