"""
Member 1: Boon
Part D & E (Base Model 1): Logistic Regression Baseline & 5-Fold GridSearchCV Tuning
Follows Group 13 Plan v2 Shared Technical Contract (Section 2.2)
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from modules.shared.utils import (
    RANDOM_STATE,
    get_data,
    build_pipeline,
    CV,
    SCORING,
    save_results,
)


def main():
    print("=" * 60)
    print("BOON - PART D & E: LOGISTIC REGRESSION (BASE MODEL 1)")
    print("=" * 60)

    # 1. Load shared 80/20 stratified split
    X_train, X_test, y_train, y_test = get_data()

    # 2. Baseline Logistic Regression pipeline (Scaler -> SMOTE -> LR)
    baseline_pipe = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )

    # 3. 5-Fold Stratified GridSearchCV over C, penalty (L1/L2), and solver
    pipe = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
    }

    print("Running 5-Fold Stratified GridSearchCV (8 candidates x 5 folds = 40 fits)...")
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=CV,
        scoring=SCORING,
        refit="f1",
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    print(f"Best Hyperparameters : {search.best_params_}")
    print(f"Best Mean 5-Fold CV F1: {search.best_score_:.4f}")

    # 4. Save results to results/logistic_regression.json
    save_results(
        "logistic_regression",
        baseline_pipe,
        search,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    print("[SUCCESS] Saved results to results/logistic_regression.json")


if __name__ == "__main__":
    main()