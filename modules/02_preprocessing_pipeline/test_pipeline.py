"""
Sanity checks for the shared data contract. Run from the repo root:
    python modules/02_preprocessing_pipeline/test_pipeline.py
(Also works with pytest if you have it installed.)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import importlib

from sklearn.linear_model import LogisticRegression

from modules.shared.utils import CONTINUOUS_COLS, build_pipeline, get_data

dp = importlib.import_module("modules.02_preprocessing_pipeline.data_pipeline")
RAW = dp.load_raw()
CLEAN = dp.clean_and_encode(RAW)
X_train, X_test, y_train, y_test = get_data()


def test_raw_shape():
    assert RAW.shape == (7043, 21)


def test_duplicates_removed():
    assert len(CLEAN) == 7043 - 22
    assert not CLEAN.duplicated().any()


def test_no_nans_no_id_all_numeric():
    for X in (CLEAN, X_train, X_test):
        assert X.isna().sum().sum() == 0
        assert "customerID" not in X.columns
        assert all(str(t).startswith(("int", "float")) for t in X.dtypes)


def test_split_shapes():
    assert (len(X_train), len(X_test)) == (5616, 1405)  # 80/20 of 7,021 rows
    assert X_train.shape[1] == X_test.shape[1] == CLEAN.shape[1] - 1
    assert set(X_train.index).isdisjoint(X_test.index)


def test_churn_ratio_preserved():
    for y in (y_train, y_test):
        assert abs(y.mean() - 0.265) < 0.01, y.mean()


def test_engineered_features():
    assert set(CONTINUOUS_COLS) <= set(X_train.columns)
    assert CLEAN["TotalServices"].between(0, 9).all()
    new = CLEAN["tenure"] == 0
    assert (CLEAN.loc[new, "AvgMonthlyCharge"] == CLEAN.loc[new, "MonthlyCharges"]).all()


def test_pipeline_never_resamples_at_predict():
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    assert list(pipe.named_steps) == ["preprocess", "smote", "model"]
    assert len(pipe.predict(X_test)) == len(X_test)


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed. Train {X_train.shape}, test {X_test.shape}")
