import sys, pathlib  # put the repo root on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import json

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from modules.shared.utils import (CONTINUOUS_COLS, CV, RANDOM_STATE, RESULTS_DIR,
                                  SCORING, get_data)

X_train, X_test, y_train, y_test = get_data()  # training set only; test set unused

corr = X_train[CONTINUOUS_COLS].corr(method="pearson")
spearman = X_train[CONTINUOUS_COLS].corr(method="spearman")

print(f"\nPearson correlation, continuous features (training set, n={len(X_train)})\n")
print(corr.round(3).to_string())

pairs = [("AvgMonthlyCharge", "MonthlyCharges"),
         ("TotalCharges", "tenure"),
         ("TotalServices", "MonthlyCharges")]
print("\nKey pairs:")
summary = {}
for a, b in pairs:
    r, rho = corr.loc[a, b], spearman.loc[a, b]
    summary[f"{a} vs {b}"] = {"pearson": float(r), "spearman": float(rho)}
    print(f"  {a + ' vs ' + b:<38} Pearson {r:.3f} | Spearman {rho:.3f}")

# Point-biserial correlation with the target (Pearson with a 0/1 variable)
churn_corr = X_train[CONTINUOUS_COLS].corrwith(y_train)
print("\nCorrelation with Churn:")
for c, v in churn_corr.items():
    print(f"  {c:<20}{v:+.3f}")

# Does AvgMonthlyCharge add information? Same CV with and without it.
# Pipeline built here (not in utils.py) because the scaled column list changes.
print("\nAblation: CV with vs without AvgMonthlyCharge (scaler -> SMOTE -> model)")
ablation = {}
for drop in [False, True]:
    Xd = X_train.drop(columns=["AvgMonthlyCharge"]) if drop else X_train
    cont = [c for c in CONTINUOUS_COLS if c in Xd.columns]
    for name, model in [
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("random_forest_tuned", RandomForestClassifier(max_depth=8, min_samples_leaf=2,
                                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ]:
        pipe = Pipeline([
            ("preprocess", ColumnTransformer([("scale", StandardScaler(), cont)],
                                             remainder="passthrough")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ])
        cv = cross_validate(pipe, Xd, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        key = f"{name} | {'without' if drop else 'with'} AvgMonthlyCharge"
        ablation[key] = {m: float(np.mean(cv[f"test_{m}"])) for m in SCORING}
        a = ablation[key]
        print(f"  {key:<55} F1 {a['f1']:.3f} | ROC-AUC {a['roc_auc']:.3f} | Recall {a['recall']:.3f}")

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "correlation_check.json", "w", encoding="utf-8") as f:
    json.dump({"pearson_matrix": corr.round(4).to_dict(),
               "key_pairs": summary,
               "corr_with_churn": churn_corr.round(4).to_dict(),
               "ablation_avg_monthly_charge": ablation}, f, indent=2)
print("\nSaved -> results/correlation_check.json")
