import sys, pathlib  # put the repo root on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import json

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from modules.shared.utils import (CV, RANDOM_STATE, REPO_ROOT, RESULTS_DIR, SCORING,
                                  build_pipeline, get_data, save_results)

MODEL_NAME = "random_forest"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def make_rf():
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


def main():
    print("=" * 60)
    print("CHANG - PART D & E: RANDOM FOREST (BASE MODEL 2)")
    print("=" * 60)

    # 1. Shared 80/20 stratified split (unscaled, not resampled)
    X_train, X_test, y_train, y_test = get_data()

    # 2. Baseline: default hyperparameters, same pipeline (scaler -> SMOTE -> RF)
    baseline_pipe = build_pipeline(make_rf())

    # 3. Grid from Plan v2.1: 2 x 3 x 2 x 2 = 24 candidates x 5 folds = 120 fits
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [2, 4],
    }

    print("Running 5-fold stratified GridSearchCV (24 candidates x 5 folds = 120 fits)...")
    search = GridSearchCV(build_pipeline(make_rf()), param_grid, cv=CV, scoring=SCORING,
                          refit="f1", n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)  # tuning uses the training set only

    # 4. Overfitting check: train vs CV F1 for the best candidate
    i = search.best_index_
    train_f1 = search.cv_results_["mean_train_f1"][i]
    cv_f1 = search.cv_results_["mean_test_f1"][i]
    print(f"Best candidate: train F1 {train_f1:.3f} | CV F1 {cv_f1:.3f} | gap {train_f1 - cv_f1:.3f}")

    # 5. Baseline vs tuned, CV + final test metrics -> results/random_forest.json
    save_results(MODEL_NAME, baseline_pipe, search, X_train, y_train, X_test, y_test)

    # 6. Feature importances (mean decrease in impurity) from the tuned forest
    best = search.best_estimator_
    names = best.named_steps["preprocess"].get_feature_names_out()
    imp = best.named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1]
    ranked = [{"feature": str(names[k]), "importance": float(imp[k])} for k in order]

    with open(RESULTS_DIR / f"{MODEL_NAME}_importances.json", "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    print("\nTop 10 feature importances (tuned RF):")
    for r in ranked[:10]:
        print(f"  {r['feature']:<40}{r['importance']:.4f}")

    top = ranked[:15][::-1]  # largest at the top of the chart
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([r["feature"] for r in top], [r["importance"] for r in top], color="#4C72B0")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest (tuned): top 15 feature importances")
    fig.tight_layout()
    fig_path = OUTPUT_DIR / "rf_feature_importances.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved -> {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
