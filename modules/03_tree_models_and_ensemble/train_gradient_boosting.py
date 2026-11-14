# Part D/E - Gradient Boosting Baseline and Tuning
# Group 13 - 6001CMD
# Uses the shared technical contract from modules/shared/utils.py.

from pathlib import Path
import json
import sys

# Allow imports from the project root when invoked as a script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

from modules.shared.utils import (
    get_data,
    build_pipeline,
    CV,
    SCORING,
    save_results,
)


def main():
    # 1. Load the team's shared 80/20 stratified split.
    X_train, X_test, y_train, y_test = get_data()

    print("=== Gradient Boosting Baseline ===")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 2. Baseline: scaling and SMOTE are inside build_pipeline(),
    #    so each CV fold fits them only on its training data.
    baseline_pipeline = build_pipeline(
        GradientBoostingClassifier(random_state=42)
    )
    baseline_scores = cross_validate(
        baseline_pipeline,
        X_train,
        y_train,
        cv=CV,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )

    print("\n=== Baseline CV Results ===")
    for metric in SCORING:
        values = baseline_scores[f"test_{metric}"]
        print(f"{metric}: {values.mean():.4f} (+/- {values.std():.4f})")
    print(f"Mean Training F1: {baseline_scores['train_f1'].mean():.4f}")

    # 3. Exactly 16 combinations: 2 x 2 x 2 x 2.
    param_grid = {
        "model__n_estimators": [100, 150],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
        "model__subsample": [0.8, 1.0],
    }
    tuned_pipeline = build_pipeline(
        GradientBoostingClassifier(random_state=42)
    )
    search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        cv=CV,
        refit="f1",
        return_train_score=True,
        n_jobs=-1,
        verbose=2,
        error_score="raise",
    )

    print("\n=== Starting GridSearchCV ===")
    search.fit(X_train, y_train)

    # 4. Print every configuration, ranked by its mean CV F1.
    print("\n=== All 16 Configurations (ranked by CV F1) ===")
    order = sorted(
        range(len(search.cv_results_["params"])),
        key=lambda i: search.cv_results_["rank_test_f1"][i],
    )
    for i in order:
        params = search.cv_results_["params"][i]
        rank = search.cv_results_["rank_test_f1"][i]
        f1 = search.cv_results_["mean_test_f1"][i]
        print(
            f"Rank {rank:2d} | F1={f1:.4f} | "
            f"trees={params['model__n_estimators']}, "
            f"rate={params['model__learning_rate']}, "
            f"depth={params['model__max_depth']}, "
            f"subsample={params['model__subsample']}"
        )

    print("\n=== Best Parameters ===")
    print(search.best_params_)
    print("\n=== Tuned CV Results ===")
    best_index = search.best_index_
    for metric in SCORING:
        value = search.cv_results_[f"mean_test_{metric}"][best_index]
        print(f"{metric}: {value:.4f}")

    # 5. Fit the baseline on all training data for the shared exporter.
    #    The tuned best_estimator_ was already refit by GridSearchCV.
    baseline_pipeline.fit(X_train, y_train)

    # Expected Plan v2 helper signature. Do not rewrite shared/utils.py.
    save_results(
        "gradient_boosting",
        baseline_pipeline,
        search,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # 6. Confirm that the expected results file exists and is parseable.
    output_path = ROOT / "results" / "gradient_boosting.json"
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Expected {output_path}; check the output path in shared save_results()."
        )
    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    required = {"model", "param_grid", "baseline", "tuned"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"JSON is missing fields {sorted(missing)}. "
            "Ask Chang to check the shared save_results() contract."
        )
    print(f"\nResults saved successfully: {output_path}")


if __name__ == "__main__":
    main()