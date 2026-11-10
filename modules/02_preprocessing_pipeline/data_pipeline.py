"""
Data cleaning and encoding for the Telco Customer Churn dataset (Part C).

This module is the single source of truth for every row-wise transformation.
modules/shared/utils.py calls clean_and_encode() inside get_data(), so every
model in the project sees exactly the same features.

Nothing in here is *fitted* on the data (no means, no scalers, no resampling),
so it is safe to run on the full dataset before the train/test split.
Anything that learns from data (StandardScaler, SMOTE) lives in
build_pipeline() in utils.py and is fitted on training folds only.

Run directly to export the processed CSV:
    python modules/02_preprocessing_pipeline/data_pipeline.py
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "telco_churn_processed.csv"

# ---------------------------------------------------------------- column groups
TARGET = "Churn"
ID_COL = "customerID"

# Services counted in TotalServices (9 in total).
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]  # InternetService is counted separately because its values are DSL/Fiber/No

# Yes/No columns mapped to 1/0 (after the "No internet/phone service" recode).
BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]

# Multi-class nominal columns -> One-Hot Encoding (drop_first=True).
ONE_HOT_COLS = ["InternetService", "Contract", "PaymentMethod"]

# Continuous / count columns that get StandardScaler inside the pipeline.
CONTINUOUS_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "TotalServices",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the untouched Kaggle CSV (7,043 x 21)."""
    return pd.read_csv(path)


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, feature-engineered, fully numeric copy of the raw data.

    Steps (order matters):
      1. TotalCharges -> numeric; the 11 blank strings become 0.0
      2. drop customerID
      3. drop exact duplicate rows (22)
      4. feature engineering: AvgMonthlyCharge, TotalServices
      5. recode 'No internet service' / 'No phone service' -> 'No'
      6. binary mapping (Yes/No, gender, target)
      7. One-Hot Encoding with drop_first=True
    """
    df = df.copy()

    # 1. Missing values ------------------------------------------------------
    # The blanks are ' ' strings, so pandas does not see them as NaN on load.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna()
    # Defensive check: our justification (nothing billed yet) only holds if
    # every blank belongs to a tenure-0 customer.
    assert (df.loc[missing, "tenure"] == 0).all(), "Blank TotalCharges with tenure > 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. Identifier carries no predictive information -----------------------
    df = df.drop(columns=[ID_COL])

    # 3. Duplicates (only detectable once the unique ID is gone) -------------
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Feature engineering (on the original labels, before recoding) ------
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)
    df["AvgMonthlyCharge"] = df["AvgMonthlyCharge"].fillna(df["MonthlyCharges"])
    df["TotalServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (
        df["InternetService"] != "No"
    ).astype(int)

    # 5. Redundant levels: InternetService / PhoneService already say "No" ---
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # 6. Binary mapping -------------------------------------------------------
    for col in BINARY_YES_NO_COLS + [TARGET]:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)

    # 7. One-Hot Encoding -----------------------------------------------------
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True, dtype=int)

    # Target last, for readability of the exported CSV.
    return df[[c for c in df.columns if c != TARGET] + [TARGET]]


def export_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Write cleaned + encoded (NOT scaled, NOT resampled) data to CSV."""
    df = clean_and_encode(load_raw())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = export_processed()
    print(f"Saved {out.shape} to {PROCESSED_PATH}")
    print(f"Churn rate after cleaning: {out[TARGET].mean():.4f}")
