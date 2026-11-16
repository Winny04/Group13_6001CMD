"""6001CMD Group 13 — Part G: Evaluation and comparison.

Run from the repository root:
    python modules/03_tree_models_and_ensemble/run_evaluation.py

The four results/*.json files are the source of truth for the summary figures.
An overlaid ROC plot additionally requires the shared get_data()/build_pipeline()
and the original raw dataset; use --skip-roc to make the summary outputs only.
No model selection or retuning occurs on the held-out test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from warnings import warn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# This script is intended to live in modules/03_tree_models_and_ensemble/.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "modules" / "03_tree_models_and_ensemble" / "outputs"
MODEL_KEYS = ("logistic_regression", "random_forest", "gradient_boosting", "ensemble")
NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ensemble": "Soft Voting Ensemble",
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_jsons():
    data = {}
    for name in MODEL_KEYS:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Obtain the teammate's results first.")
        with path.open(encoding="utf-8") as file:
            item = json.load(file)
        if item.get("model") != name:
            raise ValueError(f"{path} contains model={item.get('model')!r}, expected {name!r}")
        for section in ("cv", "test"):
            if section not in item.get("tuned", {}):
                raise ValueError(f"{path}: missing tuned.{section}")
        for metric in METRICS:
            if metric not in item["tuned"]["test"]:
                raise ValueError(f"{path}: missing tuned.test.{metric}")
        data[name] = item
    return data


def audit(data):
    print("=== Data / environment checks ===")
    for name, item in data.items():
        meta = item.get("meta", {})
        print(f"{NAMES[name]:21s} train={meta.get('n_train')} "
              f"test={meta.get('n_test')} features={meta.get('n_features')} "
              f"seed={meta.get('random_state')} "
              f"Python={meta.get('python', 'not saved')} "
              f"sklearn={meta.get('sklearn', 'not saved')}")
        cm = np.asarray(item["tuned"]["test"].get("confusion_matrix", []))
        if cm.shape != (2, 2):
            raise ValueError(f"Bad confusion matrix in {name}")
        if "n_test" in meta and int(cm.sum()) != meta["n_test"]:
            raise ValueError(f"{name}: confusion-matrix total differs from n_test")
    for field in ("n_train", "n_test", "n_features", "random_state"):
        values = {x.get("meta", {}).get(field) for x in data.values() if field in x.get("meta", {})}
        if len(values) > 1:
            raise ValueError(f"JSONs disagree on {field}: {values}")
    sklearn_versions = {x.get("meta", {}).get("sklearn") for x in data.values() if x.get("meta", {}).get("sklearn")}
    if len(sklearn_versions) > 1:
        warn("Different scikit-learn versions are recorded in the JSONs. "
             "Confirm pinned requirements and shared preprocessing before final submission.")
    print()


def export_metrics(data):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MODEL_KEYS:
        item = data[name]["tuned"]
        cv = item["cv"]
        test = item["test"]
        rows.append({
            "model": NAMES[name],
            "cv_f1": cv["f1"],
            "cv_roc_auc": cv["roc_auc"],
            **{f"test_{m}": test[m] for m in METRICS},
            "train_f1": item.get("train_f1", ""),
            "train_minus_test_f1": (item["train_f1"] - test["f1"] if "train_f1" in item else ""),
            "fit_time_s": item.get("fit_time_s", ""),
            "predict_ms_per_1000": item.get("predict_ms_per_1000", ""),
        })
    path = OUTPUTS / "model_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("=== Tuned model performance ===")
    print(f"{'Model':22s} {'CV F1':>7} {'Test F1':>8} {'Recall':>8} {'ROC-AUC':>8}")
    for row in rows:
        print(f"{row['model']:22s} {row['cv_f1']:7.4f} {row['test_f1']:8.4f} "
              f"{row['test_recall']:8.4f} {row['test_roc_auc']:8.4f}")
    print(f"Saved: {path}\n")
    return rows


def performance_chart(data):
    labels = [NAMES[k] for k in MODEL_KEYS]
    xs = np.arange(len(labels))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    width = 0.15
    for i, metric in enumerate(metrics):
        vals = [data[k]["tuned"]["test"][metric] for k in MODEL_KEYS]
        ax.bar(xs + (i - 2) * width, vals, width, label=metric.replace("_", " ").upper())
    ax.set_ylim(0, 1)
    ax.set_xticks(xs, labels, rotation=7)
    ax.set_ylabel("Held-out test score")
    ax.set_title("Tuned models: held-out test performance")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = OUTPUTS / "performance_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def confusion_charts(data):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    for key, ax in zip(MODEL_KEYS, axes.ravel()):
        cm = np.asarray(data[key]["tuned"]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=15, color="white" if val > cm.max() * .6 else "black")
        ax.set_title(NAMES[key])
        ax.set_xticks([0, 1], ["Stay (0)", "Churn (1)"])
        ax.set_yticks([0, 1], ["Stay (0)", "Churn (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices: tuned models (hold-out set)", fontsize=14)
    fig.tight_layout()
    path = OUTPUTS / "confusion_matrices.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def f1_gap_chart(data):
    xs = np.arange(len(MODEL_KEYS))
    labels = [NAMES[k] for k in MODEL_KEYS]
    train = [data[k]["tuned"]["train_f1"] for k in MODEL_KEYS]
    test = [data[k]["tuned"]["test"]["f1"] for k in MODEL_KEYS]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(xs - .17, train, .34, label="Training F1")
    ax.bar(xs + .17, test, .34, label="Test F1")
    ax.set_xticks(xs, labels, rotation=7)
    ax.set_ylim(0, .85)
    ax.set_ylabel("F1-score")
    ax.set_title("Training vs test F1: potential overfitting")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = OUTPUTS / "train_test_f1.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def timing_chart(data):
    labels = [NAMES[k] for k in MODEL_KEYS]
    xs = np.arange(len(labels))
    fit = [data[k]["tuned"]["fit_time_s"] for k in MODEL_KEYS]
    pred = [data[k]["tuned"]["predict_ms_per_1000"] for k in MODEL_KEYS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    a.bar(xs, fit)
    a.set_title("Training time (recorded)")
    a.set_ylabel("Seconds")
    b.bar(xs, pred)
    b.set_title("Prediction latency (recorded)")
    b.set_ylabel("Milliseconds per 1,000 rows")
    for ax in (a, b):
        ax.set_xticks(xs, ["LR", "RF", "GB", "Ensemble"])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Efficiency — measured in different runs/environments")
    fig.tight_layout()
    path = OUTPUTS / "execution_times.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def paired_cv_comparison(data):
    # Descriptive comparison of matching folds, NOT a p-value or significance claim.
    top = "random_forest"  # Highest individual tuned CV F1 in these files.
    ensemble = np.asarray(data["ensemble"]["tuned"]["cv_folds"]["f1"], dtype=float)
    rf = np.asarray(data[top]["tuned"]["cv_folds"]["f1"], dtype=float)
    if len(ensemble) != len(rf):
        raise ValueError("Models have different numbers of CV folds")
    diffs = ensemble - rf
    path = OUTPUTS / "paired_cv_f1.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "ensemble_f1", "random_forest_f1", "ensemble_minus_rf"])
        for i, (e, r, d) in enumerate(zip(ensemble, rf, diffs), start=1):
            writer.writerow([i, e, r, d])
    print(f"Paired five-fold CV F1 differences, Ensemble minus RF: {[round(v, 4) for v in diffs]}")
    print(f"Mean difference: {diffs.mean():+.5f}; this alone does not establish statistical significance.")
    print("Saved:", path, "\n")


def rebuild_base(name, data):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        
    "random_forest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    model = models[name]
    params = {k.removeprefix("model__"): v for k, v in data[name]["tuned"]["best_params"].items()}
    return model.set_params(**params)


def roc_chart(data):
    # Must refit on training data to obtain scores: JSONs do not contain ROC points.
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    sys.path.insert(0, str(ROOT))
    from modules.shared.utils import get_data, build_pipeline
    X_train, X_test, y_train, y_test = get_data()
    for name in MODEL_KEYS:
        meta = data[name].get("meta", {})
        for key, value in (("n_train", len(X_train)), ("n_test", len(X_test)),
                           ("n_features", X_train.shape[1]), ("random_state", 42)):
            if key in meta and meta[key] != value:
                raise ValueError(f"{name} {key}={meta[key]} differs from current {value}")

    fig, ax = plt.subplots(figsize=(7.7, 6.3))
    for key in MODEL_KEYS:
        if key == "ensemble":
            names = data["ensemble"]["selection"]["selected_models"]
            members = [(name, rebuild_base(name, data)) for name in names]
            estimator = VotingClassifier(estimators=members, voting="soft")
        else:
            estimator = rebuild_base(key, data)
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)
        observed_cm = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
        expected_cm = data[key]["tuned"]["test"]["confusion_matrix"]
        computed_auc = roc_auc_score(y_test, probabilities)
        expected_auc = data[key]["tuned"]["test"]["roc_auc"]
        # Rebuilt predictions need to match the recorded run before presenting a
        # ROC plot alongside the stored JSON metrics.
        if observed_cm != expected_cm or abs(computed_auc - expected_auc) > .002:
            plt.close(fig)
            raise RuntimeError(
                f"Rebuilt {NAMES[key]} differs from saved test results: "
                f"confusion matrix {observed_cm} vs {expected_cm}, "
                f"AUC {computed_auc:.5f} vs {expected_auc:.5f}. "
                "Check preprocessing and pinned library versions; do not mix these outputs."
            )
        fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)
        ax.plot(fpr, tpr, lw=2, label=f"{NAMES[key]} (AUC {computed_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1.2, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — all four tuned models")
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = OUTPUTS / "roc_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser(description="Summarize tuned-model results for Part G")
    parser.add_argument("--skip-roc", action="store_true",
                        help="Generate figures available from JSON without fitting models")
    args = parser.parse_args()
    data = load_jsons()
    audit(data)
    export_metrics(data)
    performance_chart(data)
    confusion_charts(data)
    f1_gap_chart(data)
    timing_chart(data)
    paired_cv_comparison(data)
    if not args.skip_roc:
        print("=== Rebuilding fixed models on training data for ROC curves ===")
        try:
            roc_chart(data)
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
            print("ROC chart not generated:", exc)
            print("Other outputs are intact. Run with --skip-roc to bypass this step.")
    else:
        print("ROC curve skipped by request; JSON files contain AUCs, not ROC coordinates.")
    print("\nCompleted. Outputs directory:", OUTPUTS)


if __name__ == "__main__":
    main()
