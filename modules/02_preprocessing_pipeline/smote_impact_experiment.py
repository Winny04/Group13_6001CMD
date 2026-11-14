"""
Part C - SMOTE impact experiment (Plan v2.1, Part C "Impact evidence").

Same model, same 5-fold stratified CV, four ways of handling class imbalance:
    none      : no resampling, no weighting (the "do nothing" reference)
    smote     : StandardScaler -> SMOTE -> model   (project default)
    balanced  : no resampling, class_weight="balanced"
    smotenc   : StandardScaler -> SMOTENC -> model (binary / one-hot columns
                treated as categorical, so no fractional values are created)

Run from the repo root:
    python modules/02_preprocessing_pipeline/smote_impact_experiment.py

Uses the TRAINING set only (CV). The test set is never touched here.
Writes results/smote_impact.json.
"""
import sys, pathlib  # put the repo root on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import json
from datetime import datetime

import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from modules.shared.utils import (CONTINUOUS_COLS, CV, RANDOM_STATE, RESULTS_DIR,
                                  SCORING, build_pipeline, get_data)

X_train, X_test, y_train, y_test = get_data()  # X_test / y_test deliberately unused


def make_model(kind, balanced=False):
    cw = "balanced" if balanced else None
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE)
    return RandomForestClassifier(class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)


def smotenc_pipeline(model):
    """Same as build_pipeline(model), but SMOTENC instead of SMOTE.

    The ColumnTransformer outputs the continuous columns first, then the
    passthrough columns, so every index from len(CONTINUOUS_COLS) onwards
    is a binary or one-hot column (categorical for SMOTENC).
    """
    base = build_pipeline(model, sampler=None)
    n_cont = len(CONTINUOUS_COLS)
    cat_idx = list(range(n_cont, X_train.shape[1]))
    return Pipeline([
        ("preprocess", base.named_steps["preprocess"]),
        ("smotenc", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
        ("model", model),
    ])


def configs(kind):
    return {
        "none": build_pipeline(make_model(kind), sampler=None),
        "smote": build_pipeline(make_model(kind), sampler="smote"),
        "balanced": build_pipeline(make_model(kind, balanced=True), sampler=None),
        "smotenc": smotenc_pipeline(make_model(kind)),
    }


results = {}
for kind in ["logistic_regression", "random_forest"]:
    results[kind] = {}
    for name, pipe in configs(kind).items():
        cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING,
                            n_jobs=-1, return_train_score=True)
        results[kind][name] = {
            "cv": {m: float(np.mean(cv[f"test_{m}"])) for m in SCORING},
            "cv_std": {m: float(np.std(cv[f"test_{m}"])) for m in SCORING},
            "cv_folds": {m: cv[f"test_{m}"].tolist() for m in SCORING},
            "train_f1_mean": float(np.mean(cv["train_f1"])),  # overfitting check
        }

# ------------------------------------------------------------------ print table
cols = ["recall", "precision", "f1", "roc_auc", "accuracy"]
print(f"\nCV on training set (5-fold stratified, n={len(X_train)}), mean +/- std\n")
print(f"{'model':<20}{'imbalance':<10}" + "".join(f"{c:>17}" for c in cols) + f"{'train_f1':>10}")
for kind, rows in results.items():
    for name, r in rows.items():
        cells = "".join(f"{r['cv'][c]:>10.3f} +/-{r['cv_std'][c]:.3f}" for c in cols)
        print(f"{kind:<20}{name:<10}{cells}{r['train_f1_mean']:>10.3f}")

# ------------------------------------------------------------------ save
RESULTS_DIR.mkdir(exist_ok=True)
path = RESULTS_DIR / "smote_impact.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"experiment": "smote_impact",
               "created": datetime.now().isoformat(timespec="seconds"),
               "random_state": RANDOM_STATE, "n_train": len(X_train),
               "results": results}, f, indent=2)
print(f"\nSaved -> results/{path.name}")
