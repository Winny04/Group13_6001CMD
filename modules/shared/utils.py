"""
Shared technical contract (Plan v2, Section 2). ONLY Chang edits this file.

Every model script uses:
    get_data()            -> X_train, X_test, y_train, y_test (unscaled, not resampled)
    build_pipeline(model) -> scaler -> SMOTE -> model   (imblearn Pipeline)
    CV, SCORING           -> identical cross-validation and metrics for everyone
    save_results(...)     -> results/<name>.json in the shared format
"""
import importlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

# The folder name starts with a digit, so a normal "import" statement cannot
# reach it; importlib can. This keeps data_pipeline.py the single source of truth.
dp = importlib.import_module("modules.02_preprocessing_pipeline.data_pipeline")

# ------------------------------------------------------------------ constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
REPO_ROOT = dp.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"
CONTINUOUS_COLS = dp.CONTINUOUS_COLS
TARGET = dp.TARGET

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}


# ------------------------------------------------------------------ data
def get_data():
    """Load -> clean -> encode -> 80/20 stratified split.

    Returns X_train, X_test, y_train, y_test as pandas objects.
    NOT scaled and NOT resampled: that happens inside build_pipeline(),
    so it is fitted on training folds only.
    """
    df = dp.clean_and_encode(dp.load_raw())
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


# ------------------------------------------------------------------ pipeline
def build_pipeline(model, sampler="smote"):
    """Wrap a model as: StandardScaler (continuous cols) -> SMOTE -> model.

    The final step is named "model", so grid keys must be "model__<param>".
    sampler: "smote" (project default) or None (for the Part C
    with/without-SMOTE impact experiment only).
    imblearn applies the sampler during fit() only; predict() never resamples.
    """
    preprocess = ColumnTransformer(
        [("scale", StandardScaler(), CONTINUOUS_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,  # keep original names for importances
    )
    steps = [("preprocess", preprocess)]
    if sampler == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    elif sampler is not None:
        raise ValueError("sampler must be 'smote' or None")
    steps.append(("model", model))
    return Pipeline(steps)


# ------------------------------------------------------------------ results
def _to_jsonable(obj):
    """Recursively convert numpy / sklearn objects into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)  # e.g. an estimator object inside params


def _test_metrics(fitted_pipe, X_test, y_test):
    y_pred = fitted_pipe.predict(X_test)
    y_prob = fitted_pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def _predict_ms_per_1000(fitted_pipe, X, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fitted_pipe.predict_proba(X)
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000 * (1000 / len(X))


def save_results(name, baseline_pipe, search, X_train, y_train, X_test, y_test):
    """Evaluate baseline and tuned pipelines and write results/<name>.json.

    baseline_pipe : unfitted build_pipeline(model_with_default_params)
    search        : a FITTED GridSearchCV / RandomizedSearchCV with refit="f1"
    The test set is only used here, for final reporting.
    """
    metric_names = list(SCORING)

    # ---- baseline: same CV, then refit on the full training set
    base_cv = cross_validate(clone(baseline_pipe), X_train, y_train,
                             cv=CV, scoring=SCORING, n_jobs=-1)
    base_fitted = clone(baseline_pipe).fit(X_train, y_train)
    baseline = {
        "params": base_fitted.named_steps["model"].get_params(),
        "cv": {m: float(np.mean(base_cv[f"test_{m}"])) for m in metric_names},
        "cv_std": {m: float(np.std(base_cv[f"test_{m}"])) for m in metric_names},
        "cv_folds": {m: base_cv[f"test_{m}"] for m in metric_names},
        "test": _test_metrics(base_fitted, X_test, y_test),
        "train_f1": f1_score(y_train, base_fitted.predict(X_train)),
    }

    # ---- tuned: read the best candidate's CV scores from the search
    res, i = search.cv_results_, search.best_index_
    n_splits = CV.get_n_splits()
    best = search.best_estimator_
    tuned = {
        "best_params": search.best_params_,
        "cv": {m: res[f"mean_test_{m}"][i] for m in metric_names},
        "cv_std": {m: res[f"std_test_{m}"][i] for m in metric_names},
        # per-fold scores: needed for the paired comparison in Part G
        "cv_folds": {m: [res[f"split{k}_test_{m}"][i] for k in range(n_splits)]
                     for m in metric_names},
        "test": _test_metrics(best, X_test, y_test),
        "train_f1": f1_score(y_train, best.predict(X_train)),
        "fit_time_s": getattr(search, "refit_time_", None),
        "predict_ms_per_1000": _predict_ms_per_1000(best, X_test),
    }

    grid = getattr(search, "param_grid", None) or getattr(search, "param_distributions", None)
    out = {
        "model": name,
        "param_grid": grid,
        "n_candidates": len(res["params"]),
        "baseline": baseline,
        "tuned": tuned,
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "random_state": RANDOM_STATE,
            "n_train": len(X_train), "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "python": platform.python_version(),
            "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(out), f, indent=2)

    print(f"\n=== {name} ===")
    print(f"Baseline CV F1 {baseline['cv']['f1']:.4f} | ROC-AUC {baseline['cv']['roc_auc']:.4f}")
    print(f"Tuned    CV F1 {tuned['cv']['f1']:.4f} | ROC-AUC {tuned['cv']['roc_auc']:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")
    return out
