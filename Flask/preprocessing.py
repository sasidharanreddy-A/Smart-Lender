"""
preprocessing.py
================
Canonical, shared preprocessing logic for the Smart Lender project.

This module is imported by **both** the training script (`Training/train_model.py`)
and the Flask backend (`Flask/app.py`). Keeping the logic in one place guarantees
that an applicant submitted through the web form is encoded in *exactly* the same
way as the rows the model was trained on.

Pipeline
--------
1. Drop the non-predictive `Loan_ID` column.
2. Fill missing values (mode for categoricals/credit-history/term, median for amount).
3. Encode categoricals:
     - Dependents: '3+' -> 3  (treated as numeric)
     - Gender / Married / Education / Self_Employed : binary maps
     - Property_Area : one-hot (drop-first) -> Property_Area_Semiurban, Property_Area_Urban
4. Cast Credit_History to int.
5. Scale the 5 numeric columns with StandardScaler.
6. Return a 12-column matrix in a FIXED column order defined by `FEATURES`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed schema -- the single source of truth for column order.
# ---------------------------------------------------------------------------
FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]

# Continuous columns that receive StandardScaler scaling.
NUMERIC_COLS = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

# Binary maps (applied AFTER missing-value imputation).
BINARY_MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
}

TARGET_MAP = {"Y": 1, "N": 0}


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with missing values filled deterministically."""
    df = df.copy()

    # Categorical / discrete columns -> most-frequent value.
    mode_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
        "Credit_History",
        "Loan_Amount_Term",
    ]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].dropna().mode()[0])

    # LoanAmount is right-skewed -> median is more robust than mean.
    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def _clean_dependents(series: pd.Series) -> pd.Series:
    """Convert the Dependents column to integers, treating '3+' as 3."""
    cleaned = series.astype(str).str.replace("+", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(int)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode a DataFrame that still has raw feature columns
    (Gender, Married, ..., Property_Area) into the 12-column numeric schema
    defined by `FEATURES`, WITHOUT scaling numeric columns yet.

    The returned DataFrame is reindexed to `FEATURES`, so any missing
    one-hot column (e.g. a single-row inference frame) is added and zero-filled.
    """
    df = impute_missing(df)

    # Dependents -> numeric.
    df["Dependents"] = _clean_dependents(df["Dependents"])

    # Binary categorical maps.
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)

    # Credit_History -> int.
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].round().astype(int)

    # Property_Area -> one-hot, dropping the first level (Rural).
    if "Property_Area" in df.columns:
        dummies = pd.get_dummies(df["Property_Area"], prefix="Property_Area", drop_first=True)
        df = pd.concat([df.drop(columns=["Property_Area"]), dummies], axis=1)

    # Enforce the canonical column order; add absent one-hot columns as 0.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES].astype(float)


def encode_target(series: pd.Series) -> pd.Series:
    """Map Loan_Status Y/N -> 1/0."""
    return series.map(TARGET_MAP).astype(int)


# ---------------------------------------------------------------------------
# Inference helper used by the Flask app
# ---------------------------------------------------------------------------
def transform_input(form: dict, scaler) -> np.ndarray:
    """
    Convert a raw HTML form payload into a single-row, scaled feature matrix.

    Parameters
    ----------
    form : dict
        Keys are the form field names (see index.html): Gender, Married,
        Dependents, Education, Self_Employed, ApplicantIncome,
        CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
        Property_Area.
    scaler : fitted StandardScaler (loaded from scale.pkl)

    Returns
    -------
    np.ndarray of shape (1, 12), ready for model.predict / predict_proba.
    """
    raw = pd.DataFrame(
        [
            {
                "Gender": form.get("Gender", "Male"),
                "Married": form.get("Married", "No"),
                "Dependents": str(form.get("Dependents", "0")),
                "Education": form.get("Education", "Graduate"),
                "Self_Employed": form.get("Self_Employed", "No"),
                "ApplicantIncome": float(form.get("ApplicantIncome", 0)),
                "CoapplicantIncome": float(form.get("CoapplicantIncome", 0)),
                "LoanAmount": float(form.get("LoanAmount", 0)),
                "Loan_Amount_Term": float(form.get("Loan_Amount_Term", 360)),
                "Credit_History": float(form.get("Credit_History", 1)),
                "Property_Area": form.get("Property_Area", "Urban"),
            }
        ]
    )

    encoded = encode_features(raw)            # (1, 12) float, unscaled
    encoded[NUMERIC_COLS] = scaler.transform(encoded[NUMERIC_COLS])
    return encoded.values
