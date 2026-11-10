"""Setup check: python main.py  (run from the repo root)."""
import pandas as pd

from modules.shared.utils import get_data
import importlib

dp = importlib.import_module("modules.02_preprocessing_pipeline.data_pipeline")

raw = dp.load_raw()
print("DEFAULT DATASET LOADED SUCCESSFULLY")
print(f"Raw shape: {raw.shape}")  # expected (7043, 21)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(raw.head())
print("\nDtypes:")
print(raw.dtypes)

X_train, X_test, y_train, y_test = get_data()
print(f"\nAfter cleaning + encoding + split:")
print(f"X_train {X_train.shape} | X_test {X_test.shape}")
print(f"Churn rate train {y_train.mean():.4f} | test {y_test.mean():.4f}")
