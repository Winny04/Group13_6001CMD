"""6001CMD Group 13 — Part F: Soft-voting ensemble.

Run from repository root after the three base-model JSON files are present.
Keep all model selection on training cross-validation scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.shared.utils import CV, SCORING, build_pipeline, get_data

RESULTS_DIR = ROOT / "results"
MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def read_model_result(name):
    path = RESULTS_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    if result.get("model") != name:
        raise ValueError(f"Unexpected model name in {path}")
    for key in ("f1", "roc_auc"):
        if key not in result["tuned"]["cv"]:
            raise ValueError(f"Missing tuned CV {key} in {path}")
    return result


def rebuild_model(name, saved_best_params):
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    estimator = estimators[name]
    params = {
        key.removeprefix("model__"): value
        for key, value in saved_best_params.items()
    }
    return estimator.set_params(**params)


def main():
    results = {name: read_model_result(name) for name in MODEL_NAMES}
    ranked_names = sorted(
        MODEL_NAMES,
        key=lambda name: (
            -results[name]["tuned"]["cv"]["f1"],
            -results[name]["tuned"]["cv"]["roc_auc"],
        ),
    )

    print("=== Model ranking (CV F1, then CV ROC-AUC) ===")
    for name in ranked_names:
        cv = results[name]["tuned"]["cv"]
        print(f"{name:22s} F1={cv['f1']:.4f}  ROC-AUC={cv['roc_auc']:.4f}")

    chosen = ranked_names[:2]
    print("\nSelected:", ", ".join(chosen))

    X_train, X_test, y_train, y_test = get_data()
    # Fail early if a model JSON was created on an incompatible split or feature set.
    for name, data in results.items():
        meta = data.get("meta", {})
        for field, current in (
            ("n_train", len(y_train)),
            ("n_test", len(y_test)),
            ("n_features", X_train.shape[1]),
            ("random_state", 42),
        ):
            if field in meta and meta[field] != current:
                raise ValueError(
                    f"{name} meta {field}={meta[field]} differs from current {current}. "
                    "Confirm the shared preprocessing contract before continuing."
                )

    members = [
        (name, rebuild_model(name, results[name]["tuned"]["best_params"]))
        for name in chosen
    ]
    ensemble = VotingClassifier(estimators=members, voting="soft")
    pipeline = build_pipeline(ensemble)  # Shared scaler and SMOTE: training folds only.

    print("\n=== Evaluating ensemble with shared 5-fold CV ===")
    fold_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=CV,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    cv_means = {}
    cv_stds = {}
    cv_folds = {}
    for metric in SCORING:
        values = fold_results[f"test_{metric}"]
        cv_means[metric] = float(values.mean())
        cv_stds[metric] = float(values.std())
        cv_folds[metric] = [float(v) for v in values]
        print(f"CV {metric:10s}: {cv_means[metric]:.4f} (+/- {cv_stds[metric]:.4f})")

    # No selection based on the test set. Evaluate once after the ensemble is fixed.
    print("\n=== Fitting final ensemble and evaluating the hold-out test set ===")
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time_s = perf_counter() - start

    predict_start = perf_counter()
    y_pred = pipeline.predict(X_test)
    prediction_time_s = perf_counter() - predict_start
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    test = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"Test {key:10s}: {test[key]:.4f}")
    print("Test confusion matrix:", test["confusion_matrix"])

    output = {
        "model": "ensemble",
        "param_grid": {},
        "baseline": None,  # A separate ensemble baseline is not defined in Plan v2.
        "selection": {
            "criterion": "highest tuned CV F1; CV ROC-AUC for ties",
            "ranked_models": ranked_names,
            "selected_models": chosen,
            "excluded_model": ranked_names[2],
            "voting": "soft",
            "base_best_params": {
                name: results[name]["tuned"]["best_params"] for name in chosen
            },
        },
        "tuned": {
            "best_params": {"voting": "soft", "selected_models": chosen},
            "cv": cv_means,
            "cv_std": cv_stds,
            "cv_folds": cv_folds,
            "test": test,
            "train_f1": float(f1_score(y_train, pipeline.predict(X_train))),
            "fit_time_s": float(fit_time_s),
            "predict_ms_per_1000": float(prediction_time_s * 1000 * 1000 / len(y_test)),
        },
        "meta": {
            "random_state": 42,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ensemble.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
